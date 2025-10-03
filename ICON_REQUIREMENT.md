# App Icon Requirement

The ModelFinder Windows Starter Pack requires an `app.ico` file for proper branding when building the executable.

## Current Status
- ❌ `app.ico` - **MISSING** (required for build)
- ✅ `app.ico.placeholder` - Present (placeholder file)

## To Complete the Setup

1. **Create or obtain a 256×256 pixel icon** in `.ico` format
2. **Save it as `app.ico`** in the project root directory
3. **The icon should represent** the ModelFinder application (e.g., 3D cube, search magnifying glass, etc.)

## Build Impact
- **With icon**: PyInstaller will embed the icon in the executable and Windows will display it
- **Without icon**: PyInstaller will use a default icon (still functional but not branded)

## Alternative
If you don't have an icon file, the build will still work but without custom branding. You can:
1. Use an online icon generator to create a simple .ico file
2. Convert an existing image to .ico format
3. Use the placeholder and build anyway (functional but not branded)

The build scripts will work either way - the icon is optional for functionality but recommended for a professional appearance.