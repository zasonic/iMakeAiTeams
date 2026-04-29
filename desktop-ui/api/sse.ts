// desktop-ui/api/sse.ts — EventSource wrapper for the sidecar's /api/events stream.
//
// EventSource can't set custom headers, but the Electron main process injects
// `Authorization: Bearer <token>` for every loopback request via a webRequest
// hook, so the URL stays clean and the token never appears in logs/history.
//
// Usage:
//   const sub = subscribeEvents({
//     chat_token: (data) => ...,
//     chat_done: (data) => ...,
//   });
//   sub.close();

import type { SidecarInfo } from "@/env";

export type EventHandler = (data: unknown) => void;

export interface EventSubscription {
  close: () => void;
}

export interface EventStreamOptions {
  /** Map of event-name → handler. Unknown event names are ignored silently. */
  handlers: Record<string, EventHandler>;
  /** Called when the stream connects (or reconnects). */
  onOpen?: () => void;
  /** Called when the stream errors out. ``closed`` is true when the
   *  EventSource has given up reconnecting (`readyState === CLOSED`). */
  onError?: (err: Event, info: { closed: boolean }) => void;
}

let currentSource: EventSource | null = null;

export function subscribeEvents(
  info: SidecarInfo,
  opts: EventStreamOptions,
): EventSubscription {
  // Tear down any previous stream — only one is supported at a time.
  closeEventStream();

  const url = `http://127.0.0.1:${info.port}/api/events`;
  const source = new EventSource(url);
  currentSource = source;

  source.onopen = () => opts.onOpen?.();
  source.onerror = (err) => {
    // EventSource auto-reconnects on transient errors (CONNECTING state).
    // Only forward "really gave up" closures so callers can wire recovery
    // (re-fetch sidecar info, re-subscribe) without flapping on every blip.
    const closed = source.readyState === EventSource.CLOSED;
    opts.onError?.(err, { closed });
  };

  // Wire one listener per registered event name. The server sends
  // `event: <name>\ndata: <json>` so the browser fires a named CustomEvent.
  for (const [name, handler] of Object.entries(opts.handlers)) {
    source.addEventListener(name, (e: MessageEvent) => {
      try {
        handler(JSON.parse(e.data));
      } catch (err) {
        console.warn(`[sse] failed to parse event ${name}:`, err);
      }
    });
  }

  return {
    close: () => {
      if (currentSource === source) currentSource = null;
      source.close();
    },
  };
}

export function closeEventStream(): void {
  if (currentSource) {
    currentSource.close();
    currentSource = null;
  }
}
