// desktop-ui/components/PromptLibraryPanel.tsx — PR 18.
//
// User-saved prompt templates. CRUD UI plus a kind filter, title/tag
// search, and an action that promotes a system_prompt template to the
// active default_system_prompt setting.

import { useEffect, useMemo, useState } from "react";

import {
  PromptTemplates,
  Settings,
  type PromptTemplate,
  type PromptTemplateKind,
} from "@/api/client";
import { t } from "@/i18n";
import { useAppStore } from "@/stores/appStore";

type KindFilter = "all" | PromptTemplateKind;

interface DraftState {
  id: string | null;
  title: string;
  body: string;
  kind: PromptTemplateKind;
  tags: string;
}

const EMPTY_DRAFT: DraftState = {
  id: null,
  title: "",
  body: "",
  kind: "snippet",
  tags: "",
};

function _kindLabel(k: PromptTemplateKind): string {
  return k === "snippet" ? t("prompts.kind.snippet") : t("prompts.kind.system_prompt");
}

function _formatUseCount(n: number): string {
  return t("prompts.use_count").replace("{count}", String(n));
}

export function PromptLibraryPanel() {
  const ready = useAppStore((s) => s.sidecarStatus?.status === "ready");
  const pushToast = useAppStore((s) => s.pushToast);
  const templates = useAppStore((s) => s.promptTemplates);
  const setPromptTemplates = useAppStore((s) => s.setPromptTemplates);
  const upsertPromptTemplate = useAppStore((s) => s.upsertPromptTemplate);
  const removePromptTemplate = useAppStore((s) => s.removePromptTemplate);

  const [kindFilter, setKindFilter] = useState<KindFilter>("all");
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [draft, setDraft] = useState<DraftState | null>(null);
  const [saving, setSaving] = useState<boolean>(false);
  const [loadError, setLoadError] = useState<string>("");

  useEffect(() => {
    if (!ready) return;
    let alive = true;
    PromptTemplates.list()
      .then((rows) => {
        if (alive) {
          setPromptTemplates(rows);
          setLoadError("");
        }
      })
      .catch((err) => {
        if (alive) {
          setLoadError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      alive = false;
    };
  }, [ready, setPromptTemplates]);

  const visible = useMemo<PromptTemplate[]>(() => {
    const q = searchTerm.trim().toLowerCase();
    return templates.filter((t) => {
      if (kindFilter !== "all" && t.kind !== kindFilter) return false;
      if (!q) return true;
      if (t.title.toLowerCase().includes(q)) return true;
      const tags = (t.tags || "").toLowerCase();
      if (tags.includes(q)) return true;
      return false;
    });
  }, [templates, kindFilter, searchTerm]);

  const openCreate = () => setDraft({ ...EMPTY_DRAFT });

  const openEdit = (row: PromptTemplate) =>
    setDraft({
      id: row.id,
      title: row.title,
      body: row.body,
      kind: row.kind,
      tags: row.tags || "",
    });

  const closeModal = () => setDraft(null);

  const save = async () => {
    if (!draft) return;
    if (!draft.title.trim() || !draft.body.trim()) {
      pushToast({
        kind: "warn",
        text: "Title and body are required.",
      });
      return;
    }
    setSaving(true);
    try {
      const payload = {
        title: draft.title.trim(),
        body: draft.body,
        kind: draft.kind,
        tags: draft.tags.trim(),
      };
      const saved = draft.id
        ? await PromptTemplates.update(draft.id, payload)
        : await PromptTemplates.create(payload);
      upsertPromptTemplate(saved);
      setDraft(null);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Save failed",
      });
    } finally {
      setSaving(false);
    }
  };

  const remove = async (row: PromptTemplate) => {
    if (!window.confirm(t("prompts.delete.confirm"))) return;
    try {
      await PromptTemplates.delete(row.id);
      removePromptTemplate(row.id);
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Delete failed",
      });
    }
  };

  const setAsDefault = async (row: PromptTemplate) => {
    try {
      await Settings.save("system_prompt", row.body);
      pushToast({ kind: "success", text: t("prompts.set_as_default.success") });
    } catch (err) {
      pushToast({
        kind: "error",
        text: err instanceof Error ? err.message : "Could not update default",
      });
    }
  };

  return (
    <div className="p-6 overflow-y-auto h-full" data-testid="prompt-library-panel">
      <header className="mb-4 flex items-start justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold">{t("prompts.title")}</h1>
          <p className="text-sm text-ink-dim">{t("prompts.subtitle")}</p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={openCreate}
          data-testid="prompt-new-button"
          disabled={!ready}
        >
          {t("prompts.new")}
        </button>
      </header>

      <div className="mb-3 flex flex-wrap gap-2 items-center">
        <KindFilterButtons value={kindFilter} onChange={setKindFilter} />
        <input
          type="text"
          className="input flex-1 min-w-[180px] text-sm"
          placeholder={t("prompts.search.placeholder")}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          data-testid="prompt-search-input"
          aria-label={t("prompts.search.placeholder")}
        />
      </div>

      {loadError && (
        <div className="rounded-md border border-err/40 bg-err/5 px-3 py-2 text-xs text-err mb-3">
          {loadError}
        </div>
      )}

      {visible.length === 0 ? (
        <div
          className="text-ink-faint text-sm"
          data-testid="prompt-empty-state"
        >
          {templates.length === 0
            ? t("prompts.empty")
            : t("prompts.empty_filtered")}
        </div>
      ) : (
        <ul className="space-y-2" data-testid="prompt-list">
          {visible.map((row) => (
            <li
              key={row.id}
              data-testid={`prompt-row-${row.id}`}
              className="card cursor-pointer hover:border-accent/40"
              onClick={() => openEdit(row)}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold truncate">{row.title}</h3>
                    <span
                      className="pill"
                      data-testid={`prompt-row-kind-${row.id}`}
                    >
                      {_kindLabel(row.kind)}
                    </span>
                  </div>
                  {row.tags && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {row.tags.split(",").map((tag) => {
                        const trimmed = tag.trim();
                        if (!trimmed) return null;
                        return (
                          <span
                            key={trimmed}
                            className="text-[11px] text-ink-dim border border-line rounded px-1.5 py-0.5"
                          >
                            {trimmed}
                          </span>
                        );
                      })}
                    </div>
                  )}
                  <div className="text-[11px] text-ink-faint mt-1">
                    {_formatUseCount(row.use_count)}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {row.kind === "system_prompt" && (
                    <button
                      type="button"
                      className="btn-ghost text-xs"
                      data-testid={`prompt-set-default-${row.id}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        setAsDefault(row);
                      }}
                    >
                      {t("prompts.set_as_default_system_prompt")}
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn-ghost text-xs text-err"
                    data-testid={`prompt-delete-${row.id}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      remove(row);
                    }}
                  >
                    {t("prompts.delete")}
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      {draft && (
        <PromptEditModal
          draft={draft}
          saving={saving}
          onChange={setDraft}
          onCancel={closeModal}
          onSave={save}
        />
      )}
    </div>
  );
}

interface KindFilterButtonsProps {
  value: KindFilter;
  onChange: (v: KindFilter) => void;
}

function KindFilterButtons({ value, onChange }: KindFilterButtonsProps) {
  const opts: Array<{ key: KindFilter; label: string }> = [
    { key: "all", label: t("prompts.filter.all") },
    { key: "snippet", label: t("prompts.filter.snippets") },
    { key: "system_prompt", label: t("prompts.filter.system_prompts") },
  ];
  return (
    <div className="inline-flex border border-line rounded-md overflow-hidden">
      {opts.map((o) => (
        <button
          key={o.key}
          type="button"
          data-testid={`prompt-filter-${o.key}`}
          onClick={() => onChange(o.key)}
          aria-pressed={value === o.key}
          className={`px-3 py-1 text-xs ${
            value === o.key
              ? "bg-accent/15 text-ink"
              : "text-ink-dim hover:bg-bg-2"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

interface PromptEditModalProps {
  draft: DraftState;
  saving: boolean;
  onChange: (d: DraftState) => void;
  onCancel: () => void;
  onSave: () => void;
}

function PromptEditModal({
  draft,
  saving,
  onChange,
  onCancel,
  onSave,
}: PromptEditModalProps) {
  const isEdit = draft.id !== null;
  return (
    <div
      className="fixed inset-0 z-30 flex items-center justify-center bg-bg-1/70"
      data-testid="prompt-edit-modal"
      role="dialog"
      aria-modal="true"
    >
      <div className="card w-full max-w-xl m-4 max-h-[90vh] overflow-y-auto">
        <header className="mb-3">
          <h2 className="text-lg font-semibold">
            {isEdit ? t("prompts.edit") : t("prompts.create")}
          </h2>
        </header>

        <div className="space-y-3">
          <div>
            <label className="label">{t("prompts.field.title")}</label>
            <input
              className="input"
              value={draft.title}
              onChange={(e) => onChange({ ...draft, title: e.target.value })}
              data-testid="prompt-edit-title"
              maxLength={100}
              autoFocus
            />
          </div>

          <div>
            <label className="label">{t("prompts.field.kind")}</label>
            <select
              className="input"
              value={draft.kind}
              onChange={(e) =>
                onChange({
                  ...draft,
                  kind: e.target.value as PromptTemplateKind,
                })
              }
              data-testid="prompt-edit-kind"
            >
              <option value="snippet">{t("prompts.kind.snippet")}</option>
              <option value="system_prompt">{t("prompts.kind.system_prompt")}</option>
            </select>
          </div>

          <div>
            <label className="label">{t("prompts.field.tags")}</label>
            <input
              className="input"
              value={draft.tags}
              onChange={(e) => onChange({ ...draft, tags: e.target.value })}
              data-testid="prompt-edit-tags"
              placeholder={t("prompts.field.tags.hint")}
            />
            <p className="text-[11px] text-ink-faint mt-1">
              {t("prompts.field.tags.hint")}
            </p>
          </div>

          <div>
            <label className="label">{t("prompts.field.body")}</label>
            <textarea
              className="input min-h-[180px] font-mono text-xs"
              value={draft.body}
              onChange={(e) => onChange({ ...draft, body: e.target.value })}
              data-testid="prompt-edit-body"
              maxLength={10_000}
            />
          </div>
        </div>

        <footer className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            className="btn-ghost"
            onClick={onCancel}
            data-testid="prompt-edit-cancel"
            disabled={saving}
          >
            {t("prompts.cancel")}
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={onSave}
            data-testid="prompt-edit-save"
            disabled={saving || !draft.title.trim() || !draft.body.trim()}
          >
            {saving ? "…" : t("prompts.save")}
          </button>
        </footer>
      </div>
    </div>
  );
}
