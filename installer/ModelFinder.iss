#define MyAppName "ModelFinder"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Base44"
#define MyAppExeName "ModelFinder.exe"

[Setup]
AppId={{C0A6C00F-7A9E-4E5E-B1E8-FA1D0C7EAB44}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={pf64}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=admin
OutputDir=..\dist
OutputBaseFilename=ModelFinder-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
; SetupIconFile=..\app.ico  ; Commented out - will use default icon if app.ico is missing

[Files]
Source: "..\dist\ModelFinder\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked