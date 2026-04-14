' ═══════════════════════════════════════════════════════════════════
'  START HERE — MyAI Agent Hub (Windows)
'  Double-click this file. No console window will appear.
' ═══════════════════════════════════════════════════════════════════

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run Chr(34) & Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\")) & "app\_launcher.bat" & Chr(34), 0, False
