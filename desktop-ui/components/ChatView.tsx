// desktop-ui/components/ChatView.tsx — chat panel with conversation list + streaming.
//
// v3 adds Power Mode: when `power_mode_enabled`, the renderer asks the
// backend to classify each message; "execution"-class messages go to
// /api/docker/execute and stream OpenClaw step events via SSE. "Chat"-class
// messages keep the v2 path intact.

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ClipboardEvent as ReactClipboardEvent,
  type DragEvent as ReactDragEvent,
} from "react";
import {
  VariableSizeList,
  type ListChildComponentProps,
} from "react-window";

import {
  Attachments,
  Chat,
  Docker,
  PromptTemplates,
  Settings,
  Voice,
  type Attachment,
  type ConversationExportFormat,
  type PromptTemplate,
  type SearchResult,
} from "@/api/client";
import { t } from "@/i18n";
import { ExecutionCard } from "@/components/ExecutionCard";
import { MessageBubble, type MessageRow } from "@/components/chat/MessageBubble";
import { MessageRenderer } from "@/components/MessageRenderer";
import { useAppStore, type PowerModeRun } from "@/stores/appStore";

interface ConversationRow {
  id: string;
  title?: string;
  updated_at?: string;
}

type ChatItem =
  | { kind: "message"; key: string; msg: MessageRow }
  | { kind: "run"; key: string; run: PowerModeRun }
  | { kind: "stream"; key: "stream"; buffer: string };

interface ChatRowData {
  items: ChatItem[];
  setRowHeight: (index: number, height: number) => void;
  approve: (taskId: string, approvalId: string, allow: boolean) => void;
  cancelRun: (taskId: string) => void;
  voiceOutputEnabled: boolean;
}

// PR 11: image input. Browser MIME types we accept for vision blocks.
// The backend mirrors this list — keep them in sync.
const IMAGE_ACCEPT = "image/png,image/jpeg,image/gif,image/webp";
const IMAGE_MIMES = new Set([
  "image/png",
  "image/jpeg",
  "image/jpg",
  "image/gif",
  "image/webp",
]);

function _isImageMime(t: string | undefined | null): boolean {
  if (!t) return false;
  const lc = t.toLowerCase();
  return IMAGE_MIMES.has(lc) || lc.startsWith("image/");
}

function _isImageAttachment(a: Attachment): boolean {
  return _isImageMime(a.mime_type);
}

export function ChatView() {
  const status = useAppStore((s) => s.sidecarStatus);
  const activeChat = useAppStore((s) => s.activeChat);
  const startChatStream = useAppStore((s) => s.startChatStream);
  const endChatStream = useAppStore((s) => s.endChatStream);
  const pushToast = useAppStore((s) => s.pushToast);
  const powerModeRuns = useAppStore((s) => s.powerModeRuns);
  const resolvePowerModeApproval = useAppStore((s) => s.resolvePowerModeApproval);
  const powerModeEnabled = useAppStore((s) => s.powerModeEnabled);
  const setPowerModeEnabled = useAppStore((s) => s.setPowerModeEnabled);
  const pendingAttachments = useAppStore((s) => s.pendingAttachments);
  const setPendingAttachments = useAppStore((s) => s.setPendingAttachments);
  const addPendingAttachment = useAppStore((s) => s.addPendingAttachment);
  const removePendingAttachment = useAppStore((s) => s.removePendingAttachment);
  // PR 18: snippet picker. Lazily hydrates the prompt-templates cache the
  // first time the user types "/" so chat sessions that never use a snippet
  // don't pay for the round-trip.
  const promptTemplates = useAppStore((s) => s.promptTemplates);
  const setPromptTemplates = useAppStore((s) => s.setPromptTemplates);
  const upsertPromptTemplate = useAppStore((s) => s.upsertPromptTemplate);
  const [promptTemplatesLoaded, setPromptTemplatesLoaded] = useState(false);
  // PR 17: voice recording state lives in the store so the StatusBar can
  // mirror the indicator without ChatView re-rendering it on every tick.
  const voiceRecording = useAppStore((s) => s.voiceRecording);
  const patchVoiceRecording = useAppStore((s) => s.patchVoiceRecording);
  const [voiceInputEnabled, setVoiceInputEnabled] = useState<boolean>(false);
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState<boolean>(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const recordingStreamRef = useRef<MediaStream | null>(null);
  const [recordingTick, setRecordingTick] = useState<number>(0);

  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [input, setInput] = useState<string>("");
  // PR 18: snippet dropdown. ``slashOpen`` is true whenever the input
  // starts with "/" — the picker filters by what comes after the slash.
  const [slashOpen, setSlashOpen] = useState<boolean>(false);
  const [slashIndex, setSlashIndex] = useState<number>(0);
  const [dragActive, setDragActive] = useState(false);
  // Tracks the in-flight Shift state on dragover so the overlay label
  // can show the "permanent" hint. The drop event itself reads e.shiftKey
  // for the actual decision so a stale dragover doesn't lie.
  const [dragShift, setDragShift] = useState(false);
  // PR 11: images are always ephemeral. When the dragged payload is an
  // image, the overlay swaps the persistence hint for an inline note
  // saying so. dataTransfer.items is the only place this is visible
  // during dragover (files isn't populated until drop on most browsers).
  const [dragHasImage, setDragHasImage] = useState(false);
  const dragCounterRef = useRef(0);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Send phase explicitly drives the Send button's disabled state and the
  // cleanup effects below. "classifying" covers the (potentially LLM-backed)
  // classify round-trip; while in that state, neither the chat-stream nor
  // Power Mode cleanup effects should fire.
  const [sendPhase, setSendPhase] = useState<
    "idle" | "classifying" | "chat" | "execution"
  >("idle");
  const [activeTaskId, setActiveTaskId] = useState<string>("");
  const [loadError, setLoadError] = useState<string>("");

  // PR 13: cross-conversation FTS5 search. ``searchQuery`` is the raw
  // input value; ``searchResults`` is the latest server response.
  // Searching never blocks the conversation list — when the query is
  // empty the standard list renders below the search input.
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string>("");

  const busy = sendPhase !== "idle";

  const responseRef = useRef<HTMLDivElement | null>(null);
  // Synchronous lock so two near-simultaneous Enter presses can't both pass
  // the `busy` guard before React re-renders with sendPhase="classifying".
  const sendLockRef = useRef(false);
  // Tracks whether the user is mid-IME composition. CJK input methods fire
  // Enter to commit a composition, which would otherwise submit the form
  // and lose the half-typed glyph.
  const composingRef = useRef(false);
  const ready = status?.status === "ready";

  // Sync the Power Mode flag from the sidecar on first ready. After that the
  // appStore owns the value — SettingsPanel updates it whenever the toggle
  // changes, so a refetch here would race with that.
  useEffect(() => {
    if (!ready) return;
    let alive = true;
    Settings.get()
      .then((s) => {
        if (!alive) return;
        setPowerModeEnabled(!!s.power_mode_enabled);
        setVoiceInputEnabled(!!s.voice_input_enabled);
        setVoiceOutputEnabled(!!s.voice_output_enabled);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [ready, setPowerModeEnabled]);

  // PR 17: re-sync the voice toggle whenever the user flips it elsewhere.
  // SettingsPanel writes the same setting; we listen for the focus event
  // rather than poll because the SSE stream doesn't surface settings
  // changes today.
  useEffect(() => {
    if (!ready) return;
    const onFocus = () => {
      Settings.get()
        .then((s) => {
          setVoiceInputEnabled(!!s.voice_input_enabled);
          setVoiceOutputEnabled(!!s.voice_output_enabled);
        })
        .catch(() => {});
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [ready]);

  // PR 17: tick a counter once a second while recording so the indicator
  // re-renders the elapsed-time display without the store doing it.
  useEffect(() => {
    if (!voiceRecording.isRecording) return;
    const id = window.setInterval(() => {
      setRecordingTick((n) => n + 1);
    }, 500);
    return () => window.clearInterval(id);
  }, [voiceRecording.isRecording]);

  // Load conversation list once the sidecar is ready.
  useEffect(() => {
    if (!ready) return;
    let alive = true;
    (async () => {
      try {
        const rows = (await Chat.list(50)) as ConversationRow[];
        if (alive) setConversations(rows);
        if (alive && rows.length && !activeId) setActiveId(rows[0].id);
      } catch (err) {
        if (alive) setLoadError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      alive = false;
    };
  }, [ready, activeId]);

  // Load messages when active conversation changes.
  useEffect(() => {
    if (!ready || !activeId) return;
    let alive = true;
    (async () => {
      try {
        const rows = (await Chat.messages(activeId)) as MessageRow[];
        if (alive) setMessages(rows);
      } catch (err) {
        if (alive) setLoadError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      alive = false;
    };
  }, [ready, activeId]);

  // PR 13: debounce the search input by 200ms so typing doesn't hammer
  // the FTS5 endpoint. An empty (whitespace-only) query short-circuits
  // to clearing results without firing a request.
  useEffect(() => {
    const trimmed = searchQuery.trim();
    if (!trimmed) {
      setSearchResults([]);
      setSearchError("");
      setSearchLoading(false);
      return;
    }
    if (!ready) return;
    let alive = true;
    const handle = window.setTimeout(async () => {
      setSearchLoading(true);
      try {
        const rows = await Chat.searchConversations(trimmed);
        if (alive) {
          setSearchResults(rows);
          setSearchError("");
        }
      } catch (err) {
        if (alive) {
          setSearchResults([]);
          setSearchError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (alive) setSearchLoading(false);
      }
    }, 200);
    return () => {
      alive = false;
      window.clearTimeout(handle);
    };
  }, [searchQuery, ready]);

  // PR 8: hydrate the chip strip when the conversation changes. Reset
  // local drag counters at the same time so a hung enter/leave pair from
  // the previous conversation doesn't carry state across.
  useEffect(() => {
    if (!ready || !activeId) return;
    let alive = true;
    setDragActive(false);
    setDragShift(false);
    dragCounterRef.current = 0;
    Attachments.list(activeId)
      .then((rows) => {
        if (alive) setPendingAttachments(activeId, rows);
      })
      .catch(() => {
        if (alive) setPendingAttachments(activeId, []);
      });
    return () => {
      alive = false;
    };
  }, [ready, activeId, setPendingAttachments]);

  // Refresh the chip strip after a chat send completes so the cleared
  // ephemeral rows disappear from the UI without a manual refresh.
  useEffect(() => {
    if (!ready || !activeId) return;
    if (sendPhase !== "idle") return;
    if (activeChat) return;
    let alive = true;
    Attachments.list(activeId)
      .then((rows) => {
        if (alive) setPendingAttachments(activeId, rows);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeChat, sendPhase, ready, activeId]);


  const newConversation = async () => {
    try {
      const { id } = await Chat.newConversation();
      setActiveId(id);
      const rows = (await Chat.list(50)) as ConversationRow[];
      setConversations(rows);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Failed to create conversation",
      });
    }
  };

  const uploadFiles = useCallback(
    async (files: FileList | File[], persist: boolean) => {
      if (!activeId) return;
      const list = Array.from(files);
      for (const f of list) {
        try {
          const result = await Attachments.upload(activeId, f, persist);
          addPendingAttachment(activeId, {
            id: result.id,
            conversation_id: activeId,
            filename: result.filename,
            mime_type: f.type,
            size_bytes: result.size_bytes,
            persist: result.persist,
            rag_doc_id: null,
            created_at: new Date().toISOString(),
          });
          pushToast({
            kind: "success",
            text: persist
              ? `Added ${result.filename} to your knowledge base`
              : `Attached ${result.filename}`,
          });
        } catch (err) {
          pushToast({
            kind: "error",
            text:
              err instanceof Error
                ? err.message
                : `Failed to attach ${f.name}`,
          });
        }
      }
    },
    [activeId, addPendingAttachment, pushToast],
  );

  const removeAttachment = useCallback(
    async (id: string) => {
      try {
        await Attachments.delete(id);
        if (activeId) removePendingAttachment(activeId, id);
      } catch (err) {
        pushToast({
          kind: "error",
          text: err instanceof Error ? err.message : "Failed to remove attachment",
        });
      }
    },
    [activeId, removePendingAttachment, pushToast],
  );

  // PR 11: detect images during the drag phase. ``dataTransfer.items`` is
  // available on dragenter/dragover (``files`` only populates after drop),
  // so this is the only place we can tell during the drag whether the
  // payload is an image and surface the "always ephemeral" inline note.
  const _dragHasImage = (dt: DataTransfer): boolean => {
    const items = dt.items;
    if (!items) return false;
    for (let i = 0; i < items.length; i += 1) {
      const it = items[i];
      if (it.kind === "file" && _isImageMime(it.type)) return true;
    }
    return false;
  };

  const onDragEnter = useCallback((e: ReactDragEvent<HTMLDivElement>) => {
    if (!e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    dragCounterRef.current += 1;
    setDragActive(true);
    const hasImage = _dragHasImage(e.dataTransfer);
    setDragHasImage(hasImage);
    setDragShift(hasImage ? false : e.shiftKey);
  }, []);

  const onDragLeave = useCallback((e: ReactDragEvent<HTMLDivElement>) => {
    if (!e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    dragCounterRef.current = Math.max(0, dragCounterRef.current - 1);
    if (dragCounterRef.current === 0) {
      setDragActive(false);
      setDragShift(false);
      setDragHasImage(false);
    }
  }, []);

  const onDragOver = useCallback((e: ReactDragEvent<HTMLDivElement>) => {
    if (!e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    const hasImage = _dragHasImage(e.dataTransfer);
    setDragHasImage(hasImage);
    setDragShift(hasImage ? false : e.shiftKey);
  }, []);

  const onDrop = useCallback(
    (e: ReactDragEvent<HTMLDivElement>) => {
      if (!e.dataTransfer.types.includes("Files")) return;
      e.preventDefault();
      const files = e.dataTransfer.files;
      dragCounterRef.current = 0;
      setDragActive(false);
      setDragShift(false);
      setDragHasImage(false);
      if (!files || !files.length || !activeId) return;
      // PR 11: split images out of the drop payload — they're always
      // ephemeral regardless of Shift, while text/pdf/etc. honor the
      // PR 8 Shift-to-persist convention.
      const arr = Array.from(files);
      const images = arr.filter((f) => _isImageMime(f.type));
      const others = arr.filter((f) => !_isImageMime(f.type));
      const persist = e.shiftKey;
      if (images.length > 0) void uploadFiles(images, false);
      if (others.length > 0) void uploadFiles(others, persist);
    },
    [activeId, uploadFiles],
  );

  // PR 11: paste a screenshot (or any clipboard image) directly into the
  // input. ClipboardEvent.clipboardData.items holds the file blobs.
  const onPaste = useCallback(
    (e: ReactClipboardEvent<HTMLTextAreaElement>) => {
      if (!activeId) return;
      const data = e.clipboardData;
      if (!data || !data.items || data.items.length === 0) return;
      const images: File[] = [];
      for (let i = 0; i < data.items.length; i += 1) {
        const it = data.items[i];
        if (it.kind !== "file") continue;
        if (!_isImageMime(it.type)) continue;
        const f = it.getAsFile();
        if (f) images.push(f);
      }
      if (images.length === 0) return;
      // Prevent the bitmap from also landing as text in the textarea
      // (Chromium would otherwise paste the image's filename as a string
      // alongside the file).
      e.preventDefault();
      void uploadFiles(images, false);
    },
    [activeId, uploadFiles],
  );

  // ── PR 17: voice recording ─────────────────────────────────────────────

  // Picks a MIME type the renderer's MediaRecorder can produce that the
  // backend's whisper-cli build can also read. Webm/Opus is what every
  // Chromium build supports for getUserMedia capture; whisper-cli accepts
  // it through ffmpeg (shipped alongside whisper.cpp) on Windows. wav is
  // a fallback for browsers that don't ship the Opus encoder.
  const _pickRecorderMime = (): string => {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/wav",
    ];
    for (const m of candidates) {
      if (
        typeof MediaRecorder !== "undefined" &&
        MediaRecorder.isTypeSupported &&
        MediaRecorder.isTypeSupported(m)
      ) {
        return m;
      }
    }
    return "";
  };

  const _stopRecordingTracks = useCallback(() => {
    const stream = recordingStreamRef.current;
    if (stream) {
      try {
        stream.getTracks().forEach((t) => t.stop());
      } catch {
        /* ignore */
      }
    }
    recordingStreamRef.current = null;
    mediaRecorderRef.current = null;
  }, []);

  const startRecording = useCallback(async () => {
    if (voiceRecording.isRecording) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      pushToast({
        kind: "error",
        text: "Microphone access isn't available in this build.",
      });
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordingStreamRef.current = stream;
      const mime = _pickRecorderMime();
      const recorder =
        mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
      recordedChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      recorder.onerror = () => {
        pushToast({
          kind: "error",
          text: "Recording failed. Check your microphone permissions.",
        });
        _stopRecordingTracks();
        patchVoiceRecording({
          isRecording: false,
          isTranscribing: false,
          recordingStartedAt: 0,
        });
      };
      recorder.onstop = async () => {
        const blob = new Blob(recordedChunksRef.current, {
          type: mime || "audio/webm",
        });
        recordedChunksRef.current = [];
        _stopRecordingTracks();
        if (blob.size === 0) {
          patchVoiceRecording({
            isRecording: false,
            isTranscribing: false,
            recordingStartedAt: 0,
          });
          return;
        }
        patchVoiceRecording({
          isRecording: false,
          isTranscribing: true,
        });
        try {
          const ext = (mime.includes("wav")
            ? "wav"
            : mime.includes("ogg")
              ? "ogg"
              : "webm");
          const result = await Voice.transcribe(blob, `clip.${ext}`);
          const text = (result.text || "").trim();
          if (text) {
            // Append rather than overwrite so the user can dictate on top
            // of an existing draft.
            setInput((prev) => (prev ? `${prev} ${text}` : text));
          } else {
            pushToast({ kind: "info", text: "No speech detected." });
          }
        } catch (err) {
          pushToast({
            kind: "error",
            text:
              err instanceof Error
                ? err.message
                : "Transcription failed",
          });
        } finally {
          patchVoiceRecording({
            isRecording: false,
            isTranscribing: false,
            recordingStartedAt: 0,
          });
        }
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      patchVoiceRecording({
        isRecording: true,
        isTranscribing: false,
        recordingStartedAt: Date.now(),
      });
    } catch (err) {
      pushToast({
        kind: "error",
        text:
          err instanceof Error
            ? err.message
            : "Could not access the microphone.",
      });
      _stopRecordingTracks();
    }
  }, [
    voiceRecording.isRecording,
    pushToast,
    patchVoiceRecording,
    _stopRecordingTracks,
  ]);

  const stopRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (!recorder) {
      patchVoiceRecording({
        isRecording: false,
        isTranscribing: false,
        recordingStartedAt: 0,
      });
      _stopRecordingTracks();
      return;
    }
    if (recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {
        /* onstop will run regardless */
      }
    }
  }, [_stopRecordingTracks, patchVoiceRecording]);

  const toggleRecording = useCallback(() => {
    if (voiceRecording.isRecording) {
      stopRecording();
    } else {
      void startRecording();
    }
  }, [voiceRecording.isRecording, startRecording, stopRecording]);

  // Tear down recording cleanly when the component unmounts (conversation
  // switch, view change). MediaStreams keep the mic LED on until they're
  // closed, so leaking one is a privacy bug.
  useEffect(() => {
    return () => {
      _stopRecordingTracks();
    };
  }, [_stopRecordingTracks]);

  // ── PR 18: slash-command snippet picker ────────────────────────────────

  // Hydrate the templates cache on first need. The PromptLibraryPanel does
  // the same on its own mount; this covers the case where the user opens a
  // snippet via slash command before they've ever opened the panel.
  const hydrateTemplates = useCallback(async () => {
    if (promptTemplatesLoaded || !ready) return;
    try {
      const rows = await PromptTemplates.list();
      setPromptTemplates(rows);
    } catch {
      /* surfaced via the panel; the picker just stays empty */
    } finally {
      setPromptTemplatesLoaded(true);
    }
  }, [promptTemplatesLoaded, ready, setPromptTemplates]);

  const slashQuery = useMemo<string | null>(() => {
    if (!slashOpen) return null;
    if (!input.startsWith("/")) return null;
    return input.slice(1).toLowerCase();
  }, [slashOpen, input]);

  const slashMatches = useMemo<PromptTemplate[]>(() => {
    if (slashQuery === null) return [];
    const snippets = promptTemplates.filter((t) => t.kind === "snippet");
    const q = slashQuery.trim();
    if (!q) return snippets.slice(0, 8);
    return snippets
      .filter((t) => {
        if (t.title.toLowerCase().includes(q)) return true;
        const tags = (t.tags || "").toLowerCase();
        return tags.includes(q);
      })
      .slice(0, 8);
  }, [slashQuery, promptTemplates]);

  // Reset the highlighted row whenever the filtered set changes so we don't
  // point past the end of the visible list.
  useEffect(() => {
    setSlashIndex((idx) => {
      if (idx < 0) return 0;
      if (idx >= slashMatches.length) return Math.max(0, slashMatches.length - 1);
      return idx;
    });
  }, [slashMatches.length]);

  const insertSnippet = useCallback(
    async (template: PromptTemplate) => {
      setInput(template.body);
      setSlashOpen(false);
      try {
        const updated = await PromptTemplates.use(template.id);
        upsertPromptTemplate(updated);
      } catch {
        /* counter bump is best-effort */
      }
    },
    [upsertPromptTemplate],
  );

  const onInputChange = useCallback(
    (value: string) => {
      setInput(value);
      const startsWithSlash = value.startsWith("/");
      if (startsWithSlash) {
        setSlashOpen(true);
        void hydrateTemplates();
      } else {
        setSlashOpen(false);
      }
    },
    [hydrateTemplates],
  );

  const onPickImages = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const onImagesPicked = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length && activeId) {
        void uploadFiles(files, false);
      }
      // Reset so picking the same file twice in a row still fires change.
      if (e.target) e.target.value = "";
    },
    [activeId, uploadFiles],
  );

  const send = async () => {
    if (!activeId || !input.trim() || busy) return;
    if (sendLockRef.current) return;
    sendLockRef.current = true;
    const text = input;
    setInput("");
    setSendPhase("classifying");
    setMessages((prev) => [
      ...prev,
      { id: `local-${Date.now()}`, role: "user", content: text },
    ]);

    let routedToExecution = false;
    if (powerModeEnabled) {
      try {
        const verdict = await Docker.classify(text, activeId);
        if (verdict.route === "execution") {
          const health = await Docker.health().catch(() => ({ ok: false } as { ok: boolean }));
          if (!health.ok) {
            // OpenClaw isn't healthy — fall back to chat with a hint.
            pushToast({
              kind: "warn",
              text: "Power Mode is enabled but OpenClaw isn't running. Falling back to chat.",
            });
          } else {
            const r = await Docker.execute(activeId, text);
            if (r.ok && r.task_id) {
              setActiveTaskId(r.task_id);
              setSendPhase("execution");
              routedToExecution = true;
            } else if (r.error) {
              pushToast({ kind: "error", text: r.error });
            }
          }
        }
      } catch (err) {
        // Classifier failure shouldn't block the user — fall back to chat.
        console.warn("classify failed:", err);
      }
    } else {
      // Power Mode off — but if the message *looks* execution-class, hint at it.
      try {
        const verdict = await Docker.classify(text, activeId);
        if (verdict.route === "execution") {
          pushToast({
            kind: "info",
            text:
              "I can do this for you if you enable Power Mode in Settings.",
          });
        }
      } catch {
        /* classifier is optional in this path */
      }
    }

    if (routedToExecution) {
      sendLockRef.current = false;
      return;
    }

    setSendPhase("chat");
    startChatStream(activeId);
    try {
      await Chat.send(activeId, text);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "chat send failed",
      });
      setSendPhase("idle");
      endChatStream();
    } finally {
      sendLockRef.current = false;
    }
  };

  // When a chat stream ends, drop busy and reload persisted messages.
  // Only fires for the chat path; Power Mode and classifying phases use their
  // own cleanup so this effect can't clear busy mid-flight.
  useEffect(() => {
    if (sendPhase !== "chat") return;
    if (activeChat) return; // chat still streaming
    setSendPhase("idle");
    // Move focus to the just-finished response so screen readers announce it
    // and keyboard users can immediately scroll/copy it.
    responseRef.current?.focus();
    if (!activeId) return;
    Chat.messages(activeId)
      .then((rows) => setMessages(rows as MessageRow[]))
      .catch(() => {});
  }, [activeChat, sendPhase, activeId]);

  // Watchdog: if the sidecar dies (or a chat_done event is lost) while we're
  // in the chat phase, reset the Send button instead of leaving it stuck.
  useEffect(() => {
    if (sendPhase !== "chat") return;
    if (ready) return;
    setSendPhase("idle");
    endChatStream();
  }, [ready, sendPhase, endChatStream]);

  // When the active Power Mode run finishes, drop busy and clear the task id
  // so a new send doesn't think the old (already-done) task is still active.
  const activeRun = activeTaskId ? powerModeRuns[activeTaskId] : null;
  useEffect(() => {
    if (sendPhase !== "execution") return;
    if (!activeTaskId) return;
    if (!activeRun) return;
    if (activeRun.done) {
      setSendPhase("idle");
      setActiveTaskId("");
    }
  }, [activeRun, activeTaskId, sendPhase]);

  const cancelActive = async () => {
    if (sendPhase === "execution" && activeTaskId) {
      try {
        await Docker.cancel(activeTaskId);
      } catch {
        /* ignore */
      }
      setActiveTaskId("");
      setSendPhase("idle");
      return;
    }
    if (sendPhase === "chat") {
      Chat.stop().catch(() => {});
      // The chat stream end effect will flip back to "idle".
      return;
    }
    // Classifying — abandon the in-flight request and reset.
    setSendPhase("idle");
  };

  const approve = async (taskId: string, approvalId: string, allow: boolean) => {
    try {
      await Docker.approve(taskId, approvalId, allow);
      resolvePowerModeApproval(taskId, approvalId);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Approval failed",
      });
    }
  };

  const streamingBuffer = activeChat?.conversationId === activeId ? activeChat.buffer : "";
  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeId),
    [conversations, activeId],
  );

  // Power Mode runs scoped to the active conversation, in order.
  const conversationRuns = useMemo(() => {
    return Object.values(powerModeRuns)
      .filter((r) => r.conversationId === activeId)
      .sort((a, b) => a.startedAt - b.startedAt);
  }, [powerModeRuns, activeId]);

  // Unified item stream so the virtualized list can render messages, Power
  // Mode runs, and the streaming preview as a single scrollable surface.
  const items = useMemo<ChatItem[]>(() => {
    const xs: ChatItem[] = messages.map((m) => ({
      kind: "message",
      key: m.id,
      msg: m,
    }));
    for (const run of conversationRuns) {
      xs.push({ kind: "run", key: `run-${run.taskId}`, run });
    }
    if (streamingBuffer) {
      xs.push({ kind: "stream", key: "stream", buffer: streamingBuffer });
    }
    return xs;
  }, [messages, conversationRuns, streamingBuffer]);

  // Per-row measured heights so VariableSizeList can render only the rows
  // that fit the viewport. Heights start as estimates and snap to the real
  // value after each row mounts via ResizeObserver.
  const sizeMapRef = useRef<Map<number, number>>(new Map());
  const listRef = useRef<VariableSizeList<ChatRowData> | null>(null);
  const ESTIMATED_ROW_HEIGHT = 96;
  const getItemSize = useCallback(
    (index: number) => sizeMapRef.current.get(index) ?? ESTIMATED_ROW_HEIGHT,
    [],
  );
  const setRowHeight = useCallback((index: number, height: number) => {
    if (sizeMapRef.current.get(index) === height) return;
    sizeMapRef.current.set(index, height);
    listRef.current?.resetAfterIndex(index);
  }, []);

  // Drop stale measurements when the active conversation changes — the same
  // index slot now points at a different message that almost certainly has a
  // different height.
  useEffect(() => {
    sizeMapRef.current.clear();
    listRef.current?.resetAfterIndex(0);
  }, [activeId]);

  // Auto-scroll to the bottom whenever the item list grows or the streaming
  // tail extends. Replaces the old scrollTo(scrollHeight) on the parent div.
  useEffect(() => {
    if (items.length === 0) return;
    listRef.current?.scrollToItem(items.length - 1, "end");
  }, [items.length, streamingBuffer]);

  const cancelRun = useCallback((taskId: string) => {
    Docker.cancel(taskId).catch(() => {});
  }, []);

  const rowData = useMemo<ChatRowData>(
    () => ({ items, setRowHeight, approve, cancelRun, voiceOutputEnabled }),
    [items, setRowHeight, approve, cancelRun, voiceOutputEnabled],
  );

  // Track the message-list area's pixel size so VariableSizeList can size
  // itself. react-window doesn't ship an AutoSizer; this matches what
  // react-virtualized-auto-sizer would provide without adding a dependency.
  const messageAreaRef = useRef<HTMLDivElement | null>(null);
  const [areaSize, setAreaSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = messageAreaRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setAreaSize({ width: r.width, height: r.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <div className="flex h-full">
      <div className="w-64 border-r border-line bg-bg-1 flex flex-col">
        <div className="p-3 border-b border-line space-y-2">
          <button className="btn-primary w-full" onClick={newConversation}>
            + New conversation
          </button>
          <input
            type="text"
            data-testid="chat-search-input"
            className="input w-full text-sm"
            placeholder="Search conversations…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.preventDefault();
                setSearchQuery("");
              }
            }}
            aria-label="Search conversations"
          />
        </div>
        <div className="flex-1 overflow-y-auto">
          {searchQuery.trim() ? (
            <SearchResultsPanel
              query={searchQuery.trim()}
              results={searchResults}
              loading={searchLoading}
              error={searchError}
              onSelect={(r) => {
                setActiveId(r.conversation_id);
                setSearchQuery("");
              }}
            />
          ) : (
            <>
              {conversations.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => setActiveId(c.id)}
                  className={`w-full text-left px-4 py-2 text-sm border-b border-line/30 ${
                    c.id === activeId
                      ? "bg-accent/10 text-ink"
                      : "text-ink-dim hover:bg-bg-2"
                  }`}
                >
                  <div className="truncate font-medium">{c.title || "Untitled"}</div>
                  <div className="text-[11px] text-ink-faint">{c.updated_at?.slice(0, 16)}</div>
                </button>
              ))}
              {!conversations.length && !loadError && (
                <div className="p-4 text-sm text-ink-faint">No conversations yet.</div>
              )}
              {loadError && (
                <div className="p-4 text-sm text-err">{loadError}</div>
              )}
            </>
          )}
        </div>
      </div>

      <div
        className="flex-1 flex flex-col min-w-0 relative"
        onDragEnter={onDragEnter}
        onDragLeave={onDragLeave}
        onDragOver={onDragOver}
        onDrop={onDrop}
        data-testid="chat-drop-target"
      >
        {activeId && (
          <div className="flex items-center justify-between border-b border-line bg-bg-1 px-4 py-2">
            <div className="text-sm font-medium text-ink truncate">
              {activeConversation?.title || "Untitled"}
            </div>
            <ExportMenu
              conversationId={activeId}
              conversationTitle={activeConversation?.title || "conversation"}
              disabled={messages.length === 0}
            />
          </div>
        )}
        <div
          ref={(el) => {
            messageAreaRef.current = el;
            responseRef.current = el;
          }}
          tabIndex={-1}
          className="flex-1 min-h-0 outline-none"
        >
          {areaSize.width > 0 && areaSize.height > 0 && (
            <VariableSizeList<ChatRowData>
              ref={listRef}
              width={areaSize.width}
              height={areaSize.height}
              itemCount={items.length}
              itemSize={getItemSize}
              itemData={rowData}
              itemKey={(index, data) => data.items[index]?.key ?? index}
              estimatedItemSize={ESTIMATED_ROW_HEIGHT}
              overscanCount={4}
            >
              {ChatListRow}
            </VariableSizeList>
          )}
        </div>

        <div className="border-t border-line p-3">
          <AttachmentChips
            attachments={pendingAttachments[activeId] ?? []}
            onRemove={removeAttachment}
          />
          {(voiceRecording.isRecording || voiceRecording.isTranscribing) && (
            <RecordingIndicator
              isRecording={voiceRecording.isRecording}
              isTranscribing={voiceRecording.isTranscribing}
              startedAt={voiceRecording.recordingStartedAt}
              tick={recordingTick}
            />
          )}
          <div className="flex gap-2 items-end">
            <button
              type="button"
              data-testid="chat-image-picker"
              className="btn-ghost px-2 py-1 text-base leading-none"
              onClick={onPickImages}
              disabled={!ready || !activeId || busy}
              title="Attach an image"
              aria-label="Attach an image"
            >
              {/* Inline SVG keeps the icon-only button accessible without
                  pulling in a new icon library. */}
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
            </button>
            {voiceInputEnabled && (
              <button
                type="button"
                data-testid="chat-mic-button"
                className={`btn-ghost px-2 py-1 text-base leading-none ${
                  voiceRecording.isRecording ? "text-err" : ""
                }`}
                onClick={toggleRecording}
                disabled={!ready || !activeId || voiceRecording.isTranscribing}
                title={
                  voiceRecording.isRecording
                    ? "Stop recording"
                    : "Record a voice message"
                }
                aria-label={
                  voiceRecording.isRecording
                    ? "Stop recording"
                    : "Record a voice message"
                }
                aria-pressed={voiceRecording.isRecording}
              >
                <svg
                  width="18"
                  height="18"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <rect x="9" y="2" width="6" height="12" rx="3" />
                  <path d="M5 10v2a7 7 0 0 0 14 0v-2" />
                  <line x1="12" y1="19" x2="12" y2="22" />
                </svg>
              </button>
            )}
            <input
              ref={fileInputRef}
              data-testid="chat-image-input"
              type="file"
              accept={IMAGE_ACCEPT}
              multiple
              className="hidden"
              onChange={onImagesPicked}
            />
            <div className="relative flex-1">
              <textarea
                data-testid="chat-input"
                className="input w-full min-h-[44px] max-h-40 resize-none"
                placeholder={
                  ready
                    ? powerModeEnabled
                      ? "Type a message… (Power Mode is on)"
                      : "Type a message…"
                    : "Waiting for backend…"
                }
                value={input}
                onChange={(e) => onInputChange(e.target.value)}
                onPaste={onPaste}
                onCompositionStart={() => {
                  composingRef.current = true;
                }}
                onCompositionEnd={() => {
                  composingRef.current = false;
                }}
                onKeyDown={(e) => {
                  // Slash dropdown navigation takes priority. Arrow keys and
                  // Tab move the highlight, Enter inserts, Esc dismisses.
                  if (slashOpen && slashMatches.length > 0) {
                    if (e.key === "ArrowDown") {
                      e.preventDefault();
                      setSlashIndex((i) => Math.min(slashMatches.length - 1, i + 1));
                      return;
                    }
                    if (e.key === "ArrowUp") {
                      e.preventDefault();
                      setSlashIndex((i) => Math.max(0, i - 1));
                      return;
                    }
                    if (e.key === "Tab") {
                      e.preventDefault();
                      const choice = slashMatches[slashIndex] ?? slashMatches[0];
                      if (choice) void insertSnippet(choice);
                      return;
                    }
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      const choice = slashMatches[slashIndex] ?? slashMatches[0];
                      if (choice) void insertSnippet(choice);
                      return;
                    }
                    if (e.key === "Escape") {
                      e.preventDefault();
                      setSlashOpen(false);
                      return;
                    }
                  }
                  if (e.key !== "Enter" || e.shiftKey) return;
                  // Don't submit while an IME composition is in flight (e.g.
                  // Japanese / Chinese / Korean input). isComposing covers both
                  // the keydown that commits a composition (which fires after
                  // compositionend on some browsers) and key 229 events.
                  if (composingRef.current || e.nativeEvent.isComposing) return;
                  e.preventDefault();
                  send();
                }}
                disabled={!ready || !activeId || busy}
              />
              {slashOpen && (
                <SnippetDropdown
                  matches={slashMatches}
                  highlighted={slashIndex}
                  onPick={insertSnippet}
                  onHover={setSlashIndex}
                />
              )}
            </div>
            <button
              className="btn-primary"
              onClick={send}
              disabled={!ready || !activeId || busy || !input.trim()}
            >
              Send
            </button>
            {busy && (
              <button className="btn-ghost" onClick={cancelActive}>
                Stop
              </button>
            )}
          </div>
        </div>
        {dragActive && activeId && (
          <div
            data-testid="chat-drop-overlay"
            className="absolute inset-0 z-20 flex items-center justify-center pointer-events-none border-2 border-dashed border-accent bg-bg-1/80 backdrop-blur-sm"
          >
            <div className="rounded-md border border-line bg-bg-2 px-6 py-4 text-center text-sm text-ink shadow-lg">
              <div className="font-semibold">
                {dragHasImage
                  ? "Drop image to attach"
                  : dragShift
                    ? "Drop to add to your knowledge base"
                    : "Drop to attach"}
              </div>
              <div
                data-testid="chat-drop-overlay-hint"
                className="text-ink-dim text-xs mt-1"
              >
                {dragHasImage
                  ? "Images are always ephemeral"
                  : "Drop to attach. Hold Shift to add to your knowledge base permanently."}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── PR 18: Slash-command snippet dropdown ──────────────────────────────────

interface SnippetDropdownProps {
  matches: PromptTemplate[];
  highlighted: number;
  onPick: (t: PromptTemplate) => void;
  onHover: (idx: number) => void;
}

function SnippetDropdown({
  matches,
  highlighted,
  onPick,
  onHover,
}: SnippetDropdownProps) {
  return (
    <div
      data-testid="chat-slash-dropdown"
      className="absolute bottom-full left-0 right-0 mb-1 z-20 max-h-60 overflow-y-auto rounded-md border border-line bg-bg-1 shadow-lg"
      role="listbox"
    >
      {matches.length === 0 ? (
        <div
          data-testid="chat-slash-empty"
          className="px-3 py-2 text-xs text-ink-faint"
        >
          {t("prompts.slash.hint")}
        </div>
      ) : (
        matches.map((m, i) => (
          <button
            key={m.id}
            type="button"
            role="option"
            aria-selected={i === highlighted}
            data-testid={`chat-slash-option-${m.id}`}
            onMouseDown={(e) => {
              // Use mousedown so the textarea doesn't lose focus before the
              // pick fires (which would close the dropdown on blur first).
              e.preventDefault();
              onPick(m);
            }}
            onMouseEnter={() => onHover(i)}
            className={`w-full text-left px-3 py-2 text-xs ${
              i === highlighted
                ? "bg-accent/15 text-ink"
                : "text-ink-dim hover:bg-bg-2"
            }`}
          >
            <div className="font-medium text-ink">{m.title}</div>
            <div className="text-[11px] text-ink-faint truncate">
              {m.body.slice(0, 80)}
            </div>
          </button>
        ))
      )}
    </div>
  );
}

// ── PR 17: Recording indicator ──────────────────────────────────────────────

interface RecordingIndicatorProps {
  isRecording: boolean;
  isTranscribing: boolean;
  startedAt: number;
  // ``tick`` forces a re-render once a second while the timer's running;
  // we read the elapsed time from Date.now() so the displayed value stays
  // accurate even if a few ticks are dropped (e.g. tab backgrounded).
  tick: number;
}

function RecordingIndicator({
  isRecording,
  isTranscribing,
  startedAt,
  // ``tick`` is intentionally unused inside the body — accepting it as a
  // prop is what triggers the re-render that recomputes ``elapsed``.
  tick: _tick,
}: RecordingIndicatorProps) {
  const elapsed = isRecording && startedAt
    ? Math.max(0, Math.floor((Date.now() - startedAt) / 1000))
    : 0;
  const mm = Math.floor(elapsed / 60);
  const ss = elapsed % 60;
  const formatted = `${mm}:${ss < 10 ? "0" : ""}${ss}`;
  return (
    <div
      data-testid="chat-recording-indicator"
      className="mb-2 flex items-center gap-2 text-xs"
      role="status"
      aria-live="polite"
    >
      {isTranscribing ? (
        <>
          <span
            aria-hidden="true"
            className="inline-block h-2 w-2 rounded-full bg-accent animate-pulse"
          />
          <span className="text-ink-dim">Transcribing…</span>
        </>
      ) : (
        <>
          <span
            aria-hidden="true"
            className="inline-block h-2 w-2 rounded-full bg-err animate-pulse"
          />
          <span className="text-ink-dim">
            Recording <span className="font-mono text-ink">{formatted}</span>
          </span>
        </>
      )}
    </div>
  );
}

// ── Cross-conversation FTS5 search results (PR 13) ──────────────────────────

interface SearchResultsPanelProps {
  query: string;
  results: SearchResult[];
  loading: boolean;
  error: string;
  onSelect: (r: SearchResult) => void;
}

function _formatTimestamp(iso: string): string {
  // ISO timestamps are sortable as-is and the "minute" prefix is enough
  // context for the result row's secondary text.
  return iso ? iso.slice(0, 16).replace("T", " ") : "";
}

// Renders a snippet that may contain <mark>…</mark> tags emitted by FTS5
// as alternating <span> and <mark> elements. Splits the string ourselves
// instead of using dangerouslySetInnerHTML so that any HTML the backend
// might leak through stays inert.
function _renderSnippet(snippet: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let rest = snippet;
  let key = 0;
  while (rest.length > 0) {
    const open = rest.indexOf("<mark>");
    if (open < 0) {
      parts.push(<span key={key++}>{rest}</span>);
      break;
    }
    if (open > 0) {
      parts.push(<span key={key++}>{rest.slice(0, open)}</span>);
    }
    const after = rest.slice(open + "<mark>".length);
    const close = after.indexOf("</mark>");
    if (close < 0) {
      // Unbalanced — render the tail as plain text and stop.
      parts.push(<span key={key++}>{after}</span>);
      break;
    }
    parts.push(
      <mark
        key={key++}
        className="bg-yellow-200/30 text-ink rounded px-0.5"
      >
        {after.slice(0, close)}
      </mark>,
    );
    rest = after.slice(close + "</mark>".length);
  }
  return parts;
}

function SearchResultsPanel({
  query,
  results,
  loading,
  error,
  onSelect,
}: SearchResultsPanelProps) {
  if (error) {
    return (
      <div
        data-testid="chat-search-error"
        className="p-4 text-sm text-err"
      >
        {error}
      </div>
    );
  }
  if (!results.length) {
    if (loading) {
      return (
        <div
          data-testid="chat-search-loading"
          className="p-4 text-sm text-ink-faint"
        >
          Searching…
        </div>
      );
    }
    return (
      <div
        data-testid="chat-search-empty"
        className="p-4 text-sm text-ink-faint"
      >
        No matches for &ldquo;{query}&rdquo;.
      </div>
    );
  }
  return (
    <div data-testid="chat-search-results">
      {results.map((r) => (
        <button
          key={r.message_id}
          type="button"
          data-testid={`chat-search-result-${r.message_id}`}
          onClick={() => onSelect(r)}
          className="w-full text-left px-4 py-2 text-sm border-b border-line/30 text-ink-dim hover:bg-bg-2"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="truncate text-[11px] uppercase tracking-wide text-ink-faint">
              {r.conversation_title || "Untitled"}
            </span>
            <span
              className={`inline-flex items-center justify-center px-1.5 h-[16px] text-[10px] rounded-full ${
                r.role === "user"
                  ? "bg-accent/15 text-accent"
                  : "bg-bg-3 text-ink-dim"
              }`}
            >
              {r.role}
            </span>
          </div>
          <div className="mt-0.5 text-ink line-clamp-2 break-words">
            {_renderSnippet(r.snippet)}
          </div>
          <div className="text-[11px] text-ink-faint mt-0.5">
            {_formatTimestamp(r.created_at)}
          </div>
        </button>
      ))}
    </div>
  );
}

// ── Attachment chip strip (PR 8) ────────────────────────────────────────────

interface AttachmentChipsProps {
  attachments: Attachment[];
  onRemove: (id: string) => void;
}

function _formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function AttachmentChips({ attachments, onRemove }: AttachmentChipsProps) {
  if (!attachments.length) return null;
  return (
    <div
      data-testid="chat-attachment-chips"
      className="flex flex-wrap gap-1.5 mb-2"
    >
      {attachments.map((a) =>
        _isImageAttachment(a) ? (
          <ImageAttachmentChip key={a.id} attachment={a} onRemove={onRemove} />
        ) : (
          <span
            key={a.id}
            data-testid={`chat-attachment-chip-${a.id}`}
            className={`inline-flex items-center gap-2 rounded-md border px-2 py-1 text-xs ${
              a.persist
                ? "border-accent/40 bg-accent/10 text-ink"
                : "border-line bg-bg-2 text-ink"
            }`}
            title={
              a.persist
                ? `${a.filename} — saved to knowledge base`
                : `${a.filename} — ephemeral (next send only)`
            }
          >
            <span className="truncate max-w-[14rem]">{a.filename}</span>
            <span className="text-ink-faint">{_formatBytes(a.size_bytes)}</span>
            {a.persist && (
              <span className="text-[10px] uppercase tracking-wide text-accent">
                Saved
              </span>
            )}
            <button
              type="button"
              data-testid={`chat-attachment-remove-${a.id}`}
              aria-label={`Remove ${a.filename}`}
              onClick={() => onRemove(a.id)}
              className="text-ink-dim hover:text-ink"
            >
              ×
            </button>
          </span>
        ),
      )}
    </div>
  );
}

// PR 11: image chip with a thumbnail. The thumbnail is fetched from the
// attachment endpoint so the chip rehydrates correctly after reload (the
// raw File object only exists in memory at upload time). Falls back to a
// generic icon if the network fetch fails.
interface ImageAttachmentChipProps {
  attachment: Attachment;
  onRemove: (id: string) => void;
}

function ImageAttachmentChip({ attachment, onRemove }: ImageAttachmentChipProps) {
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let revoke: string | null = null;
    let alive = true;
    Attachments.fetchBlob(attachment.id)
      .then((blob) => {
        if (!alive) return;
        const url = URL.createObjectURL(blob);
        revoke = url;
        setSrc(url);
      })
      .catch(() => {
        /* leave src null — fallback marker shows */
      });
    return () => {
      alive = false;
      if (revoke) URL.revokeObjectURL(revoke);
    };
  }, [attachment.id]);

  return (
    <span
      data-testid={`chat-attachment-chip-${attachment.id}`}
      data-image="true"
      className="inline-flex items-center gap-2 rounded-md border border-line bg-bg-2 px-2 py-1 text-xs text-ink"
      title={`${attachment.filename} — image (ephemeral)`}
    >
      <span
        className="block h-7 w-7 rounded border border-line bg-bg-1 overflow-hidden flex items-center justify-center"
      >
        {src ? (
          <img
            data-testid={`chat-attachment-thumb-${attachment.id}`}
            src={src}
            alt=""
            className="h-full w-full object-cover"
          />
        ) : (
          <span aria-hidden="true" className="text-ink-faint">🖼️</span>
        )}
      </span>
      <span className="truncate max-w-[10rem]">{attachment.filename}</span>
      <span className="text-ink-faint">{_formatBytes(attachment.size_bytes)}</span>
      <button
        type="button"
        data-testid={`chat-attachment-remove-${attachment.id}`}
        aria-label={`Remove ${attachment.filename}`}
        onClick={() => onRemove(attachment.id)}
        className="text-ink-dim hover:text-ink"
      >
        ×
      </button>
    </span>
  );
}

// ── Virtualized row renderer ────────────────────────────────────────────────

function ChatListRow({ index, style, data }: ListChildComponentProps<ChatRowData>) {
  const item = data.items[index];
  const innerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r && r.height > 0) data.setRowHeight(index, r.height);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [index, data]);

  if (!item) return null;

  return (
    <div style={style}>
      <div ref={innerRef} className="px-6 py-1.5">
        {item.kind === "message" && (
          <MessageBubble
            msg={item.msg}
            voiceOutputEnabled={data.voiceOutputEnabled}
          />
        )}
        {item.kind === "run" && (
          <PowerModeMessage
            run={item.run}
            onApprove={(approvalId, allow) =>
              data.approve(item.run.taskId, approvalId, allow)
            }
            onCancel={() => data.cancelRun(item.run.taskId)}
          />
        )}
        {item.kind === "stream" && (
          <div
            aria-live="polite"
            aria-atomic="false"
            className="max-w-[80%] rounded-xl px-4 py-2 text-sm bg-bg-2 text-ink border border-line"
          >
            {/* Don't expose the speaker on the still-streaming buffer; the
                speaker would synthesize a partial sentence. The persisted
                MessageBubble below picks it up after chat_done. */}
            <MessageRenderer content={item.buffer} role="assistant" />
            <span
              role="status"
              aria-label="Assistant is thinking"
              className="inline-block ml-1 h-3 w-1 bg-accent animate-pulse align-middle"
            />
          </div>
        )}
      </div>
    </div>
  );
}

// MessageBubble extracted to components/chat/MessageBubble.tsx (Layer C2).

// ── Power Mode message bubble ───────────────────────────────────────────────

interface PowerModeMessageProps {
  run: PowerModeRun;
  onApprove: (approvalId: string, allow: boolean) => void;
  onCancel: () => void;
}

function PowerModeMessage({ run, onApprove, onCancel }: PowerModeMessageProps) {
  const elapsed = Math.max(0, Math.floor((Date.now() - run.startedAt) / 1000));
  const showProgress = !run.done;

  return (
    <div className="max-w-[85%] rounded-xl px-4 py-3 text-sm bg-bg-2 text-ink border border-line space-y-2">
      <div className="flex items-center gap-2 text-[11px] text-ink-faint">
        <span className="px-1.5 py-0.5 rounded bg-accent/15 text-accent border border-accent/30 font-semibold">
          ⚡ Power Mode
        </span>
        <span className="font-mono">{run.taskId}</span>
        {showProgress && (
          <span className="text-warn">working… {elapsed}s</span>
        )}
        {showProgress && (
          <button type="button" className="ml-auto text-ink-dim hover:text-ink" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>

      {run.steps.length > 0 && (
        <div className="space-y-1.5">
          {run.steps.map((step) => (
            <ExecutionCard key={step.step_id} step={step} />
          ))}
        </div>
      )}

      {run.approvals.map((appr) => (
        <ApprovalCard
          key={appr.approval_id}
          summary={appr.summary}
          details={appr.details}
          danger={appr.danger}
          expiresAt={appr.expires_at}
          onAllow={() => onApprove(appr.approval_id, true)}
          onDeny={() => onApprove(appr.approval_id, false)}
        />
      ))}

      {run.resultText && (
        <div className="whitespace-pre-wrap text-sm">{run.resultText}</div>
      )}

      {run.error && (
        <div className="rounded-md border border-err/40 bg-err/5 px-3 py-2 text-err text-xs">
          {run.error}
        </div>
      )}

      {run.done && !run.error && !run.resultText && (
        <div className="text-[11px] text-ink-faint">Cancelled by user</div>
      )}
    </div>
  );
}

interface ApprovalCardProps {
  summary: string;
  details: Record<string, unknown>;
  danger: "low" | "medium" | "high";
  expiresAt: number;
  onAllow: () => void;
  onDeny: () => void;
}

function ApprovalCard({
  summary,
  details,
  danger,
  expiresAt,
  onAllow,
  onDeny,
}: ApprovalCardProps) {
  const [, force] = useState(0);
  const remaining = Math.max(0, Math.ceil((expiresAt - Date.now()) / 1000));
  useEffect(() => {
    // Run a single 1s timer per approval and stop ticking once the deadline
    // has passed. The effect re-runs only when expiresAt changes, so we
    // don't accumulate timers across re-renders.
    if (Date.now() >= expiresAt) return;
    const id = window.setInterval(() => {
      force((n) => n + 1);
      if (Date.now() >= expiresAt) window.clearInterval(id);
    }, 1000);
    return () => window.clearInterval(id);
  }, [expiresAt]);
  const tone = danger === "high"
    ? "border-err/50 bg-err/5"
    : danger === "low"
      ? "border-line bg-bg-1"
      : "border-warn/40 bg-warn/5";

  return (
    <div className={`rounded-md border ${tone} px-3 py-2 text-sm space-y-2`}>
      <div className="flex items-center justify-between">
        <span className="font-semibold">Approval needed</span>
        <span className="text-[11px] text-ink-faint">
          auto-deny in {remaining}s
        </span>
      </div>
      <p className="text-ink">{summary || "OpenClaw wants to perform an action."}</p>
      {Object.keys(details).length > 0 && (
        <pre className="text-[11px] font-mono whitespace-pre-wrap text-ink-dim border-t border-line/60 pt-2">
          {JSON.stringify(details, null, 2)}
        </pre>
      )}
      <div className="flex gap-2 pt-1">
        <button type="button" className="btn-primary text-xs" onClick={onAllow}>
          Allow
        </button>
        <button type="button" className="btn-ghost text-xs" onClick={onDeny}>
          Deny
        </button>
      </div>
    </div>
  );
}

// ── Export menu (PR 7) ──────────────────────────────────────────────────────
//
// Three formats served off the backend export routes:
//   - .md and .json land as text and are written via the existing
//     saveFileDialog IPC (renderer never touches the filesystem).
//   - .pdf-html lands as HTML and is handed to electronAPI.exportPdf, which
//     spawns a hidden BrowserWindow in main and runs printToPDF.
//
// We don't preload the export content; nothing gets fetched until the user
// picks a format from the dropdown. The button is `disabled` when the
// conversation has no messages — exporting an empty file is just noise.

interface ExportMenuProps {
  conversationId: string;
  conversationTitle: string;
  disabled?: boolean;
}

function _safeFilename(title: string): string {
  // Match the same character classes the backend orchestrator uses for
  // export filenames. The Save dialog's default name is what the user sees,
  // so a clean stem matters more than perfect parity with the backend.
  const cleaned = (title || "conversation")
    .replace(/[^A-Za-z0-9 _-]+/g, "_")
    .trim()
    .slice(0, 60);
  return cleaned || "conversation";
}

function ExportMenu({
  conversationId,
  conversationTitle,
  disabled,
}: ExportMenuProps) {
  const pushToast = useAppStore((s) => s.pushToast);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const wrapperRef = useRef<HTMLDivElement | null>(null);

  // Close the dropdown when the user clicks anywhere outside of it.
  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (!wrapperRef.current) return;
      if (!wrapperRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  const stem = _safeFilename(conversationTitle);

  const runExport = async (format: ConversationExportFormat) => {
    setOpen(false);
    setBusy(true);
    try {
      const body = await Chat.exportConversation(conversationId, format);
      if (format === "pdf-html") {
        const result = await window.electronAPI.exportPdf(body, `${stem}.pdf`);
        if (result.ok) {
          pushToast({ kind: "success", text: `Saved PDF to ${result.path}` });
        } else if (!result.cancelled) {
          pushToast({
            kind: "error",
            text: result.error || "PDF export failed",
          });
        }
      } else {
        const ext = format === "md" ? "md" : "json";
        const result = await window.electronAPI.saveFileDialog(
          `${stem}.${ext}`,
          body,
        );
        if (result.ok) {
          pushToast({ kind: "success", text: `Saved to ${result.path}` });
        } else if (!result.cancelled) {
          pushToast({
            kind: "error",
            text: result.error || "Export failed",
          });
        }
      }
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Export failed",
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div ref={wrapperRef} className="relative">
      <button
        type="button"
        data-testid="chat-export-button"
        className="btn-ghost text-xs"
        onClick={() => setOpen((o) => !o)}
        disabled={disabled || busy}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        Export {open ? "▴" : "▾"}
      </button>
      {open && (
        <div
          role="menu"
          data-testid="chat-export-menu"
          className="absolute right-0 mt-1 z-10 min-w-[12rem] rounded-md border border-line bg-bg-1 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            data-testid="chat-export-md"
            className="block w-full text-left px-3 py-2 text-xs text-ink hover:bg-bg-2"
            onClick={() => runExport("md")}
          >
            Markdown (.md)
          </button>
          <button
            type="button"
            role="menuitem"
            data-testid="chat-export-json"
            className="block w-full text-left px-3 py-2 text-xs text-ink hover:bg-bg-2"
            onClick={() => runExport("json")}
          >
            JSON (.json)
          </button>
          <button
            type="button"
            role="menuitem"
            data-testid="chat-export-pdf"
            className="block w-full text-left px-3 py-2 text-xs text-ink hover:bg-bg-2"
            onClick={() => runExport("pdf-html")}
          >
            PDF (.pdf)
          </button>
        </div>
      )}
    </div>
  );
}
