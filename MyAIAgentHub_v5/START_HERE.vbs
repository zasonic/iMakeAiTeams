' ═══════════════════════════════════════════════════════════════════
'  START HERE — MyAI Agent Hub (Windows)
'  Double-click this file. No console window will appear.
'  This VBS wrapper launches the batch file silently.
' ═══════════════════════════════════════════════════════════════════

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run Chr(34) & Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\")) & "START_HERE_Windows.bat" & Chr(34), 0, False
