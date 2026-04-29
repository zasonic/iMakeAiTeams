; installer.nsh — custom NSIS hooks injected via electron-builder.yml
; (nsis.include). Currently used to surface what user data stays on
; uninstall — see #39.
;
; deleteAppDataOnUninstall is intentionally false on this project so a
; user reinstalling (e.g. after a "Repair" or version bump) doesn't lose
; conversations, settings, and (importantly) the OS-keyring-stored API
; key sentinel. The macro below tells the user where the data lives so
; they can clean up manually if they really want a from-scratch removal.

!macro customUnInstall
    MessageBox MB_OK "iMakeAiTeams was uninstalled.$\r$\n$\r$\nYour conversations, settings, and any cached state remain in:$\r$\n  $APPDATA\iMakeAiTeams$\r$\n  $LOCALAPPDATA\iMakeAiTeams\MyAIAgentHub$\r$\n$\r$\nAPI keys stored in the Windows keyring are NOT removed by this uninstaller. Use 'credentials manager' (Windows Settings > Accounts > Credentials) and remove entries under 'iMakeAiTeams' if you want a clean removal.$\r$\n$\r$\nDelete those folders manually for a complete reset."
!macroend
