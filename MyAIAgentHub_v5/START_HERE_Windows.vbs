' ═══════════════════════════════════════════════════════════════════
'  START HERE — MyAI Agent Hub (Windows)
'  Double-click this file. No console window will appear.
' ═══════════════════════════════════════════════════════════════════

Set FSO = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

' Get paths
ScriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
AppDir = FSO.BuildPath(ScriptDir, "app")
Launcher = FSO.BuildPath(AppDir, "setup_launcher.pyw")

' ── 1. Check for portable Python ──
PortablePy  = FSO.BuildPath(AppDir, ".python\pythonw.exe")
PortablePy2 = FSO.BuildPath(AppDir, ".python\python.exe")

If FSO.FileExists(PortablePy) Then
    WshShell.Run Chr(34) & PortablePy & Chr(34) & " " & Chr(34) & Launcher & Chr(34), 0, False
    WScript.Quit
End If
If FSO.FileExists(PortablePy2) Then
    WshShell.Run Chr(34) & PortablePy2 & Chr(34) & " " & Chr(34) & Launcher & Chr(34), 0, False
    WScript.Quit
End If

' ── 2. Check for system pythonw / python ──
' Try pythonw first (no console window)
On Error Resume Next

' Check if pythonw is available
WshShell.Run "cmd /c where pythonw >nul 2>&1", 0, True
If Err.Number = 0 Then
    WshShell.Run "pythonw " & Chr(34) & Launcher & Chr(34), 0, False
    WScript.Quit
End If
Err.Clear

' Check if py launcher is available (standard on modern Windows Python)
WshShell.Run "cmd /c where py >nul 2>&1", 0, True
If Err.Number = 0 Then
    WshShell.Run "py -3 " & Chr(34) & Launcher & Chr(34), 0, False
    WScript.Quit
End If
Err.Clear

' Check if python3 is available
WshShell.Run "cmd /c where python3 >nul 2>&1", 0, True
If Err.Number = 0 Then
    WshShell.Run "python3 " & Chr(34) & Launcher & Chr(34), 0, False
    WScript.Quit
End If
Err.Clear

' Check if python is available
WshShell.Run "cmd /c where python >nul 2>&1", 0, True
If Err.Number = 0 Then
    WshShell.Run "python " & Chr(34) & Launcher & Chr(34), 0, False
    WScript.Quit
End If
Err.Clear

On Error GoTo 0

' ── 3. No Python found — run the bootstrap bat (will show console briefly) ──
BatFile = FSO.BuildPath(AppDir, "_bootstrap_windows.bat")
If FSO.FileExists(BatFile) Then
    WshShell.Run Chr(34) & BatFile & Chr(34), 1, False
Else
    MsgBox "Python is not installed." & vbCrLf & vbCrLf & _
           "Please install Python 3.10 or newer from:" & vbCrLf & _
           "https://www.python.org/downloads/" & vbCrLf & vbCrLf & _
           "Make sure to check 'Add Python to PATH' during installation.", _
           vbExclamation, "MyAI Agent Hub"
End If
