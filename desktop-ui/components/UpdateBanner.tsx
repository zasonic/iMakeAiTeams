// desktop-ui/components/UpdateBanner.tsx — non-blocking "update is ready" prompt.
//
// Listens for the IPC "update:downloaded" event (re-emitted by the preload
// bridge as `onUpdateDownloaded`) and stages a banner at the top of the app.
// The user can apply the update by clicking "Restart now" (forwards to
// `installUpdate()` which calls autoUpdater.quitAndInstall) or postpone with
// "Later" — postponement is session-scoped only, so the banner reappears on
// the next launch if the update is still pending.

import { useEffect } from "react";

import { useAppStore } from "@/stores/appStore";

export function UpdateBanner() {
  const updateReady = useAppStore((s) => s.updateReady);
  const setUpdateReady = useAppStore((s) => s.setUpdateReady);
  const updateBannerDismissed = useAppStore((s) => s.updateBannerDismissed);
  const setUpdateBannerDismissed = useAppStore((s) => s.setUpdateBannerDismissed);

  useEffect(() => {
    // window.electronAPI is only present in the Electron renderer; in tests
    // and on web previews we leave the subscription alone so the banner
    // stays driven purely by appStore state.
    const api = window.electronAPI;
    if (!api?.onUpdateDownloaded) return;
    const unsub = api.onUpdateDownloaded((info) => {
      setUpdateReady({ version: info.version });
    });
    return unsub;
  }, [setUpdateReady]);

  if (!updateReady || updateBannerDismissed) return null;

  const installNow = () => {
    window.electronAPI?.installUpdate?.();
  };

  const dismiss = () => {
    setUpdateBannerDismissed(true);
  };

  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="update-banner"
      className="flex items-center justify-between gap-3 px-4 py-2 border-b border-accent/40 bg-accent/10 text-sm text-ink"
    >
      <span className="truncate">
        Version {updateReady.version} is ready. Restart to apply.
      </span>
      <div className="flex items-center gap-2 flex-shrink-0">
        <button
          type="button"
          className="btn-primary text-xs"
          onClick={installNow}
        >
          Restart now
        </button>
        <button
          type="button"
          className="btn-ghost text-xs"
          onClick={dismiss}
        >
          Later
        </button>
      </div>
    </div>
  );
}
