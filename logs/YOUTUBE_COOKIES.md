# YouTube cookies (Edge / Windows)

YouTube often blocks anonymous `yt-dlp` access. On modern Edge/Chrome,
`--cookies-from-browser edge` frequently fails with:

```text
Failed to decrypt with DPAPI
```

That is a Windows encryption limitation, not a VisionFSD bug. Use a
**cookies file** instead.

## Recommended: export cookies from Edge

1. Install an Edge extension that exports **Netscape** `cookies.txt`, e.g.
   - **Get cookies.txt LOCALLY**
2. Open [https://www.youtube.com](https://www.youtube.com) and make sure you
   are signed in.
3. Export cookies for `youtube.com`.
4. Save the file as:

   ```text
   D:\VisionFSD-Pilot\logs\youtube-cookies.txt
   ```

5. Run `run_youtube_test.bat` again.  
   The bat **auto-detects** `logs\youtube-cookies.txt`.

## Alternative: Firefox browser cookies

Firefox cookies usually work with yt-dlp (no Chromium DPAPI issue):

1. Install Firefox, log into YouTube there.
2. Fully quit Firefox.
3. In `run_youtube_test.bat`, set:

   ```bat
   set "YT_COOKIES=--cookies-from-browser firefox"
   ```

## Manual CLI

```bat
.venv\Scripts\python.exe src\visionfsd_3d.py ^
  --source "https://www.youtube.com/watch?v=JS5HvyvhhxM" ^
  --cookies logs\youtube-cookies.txt ^
  --start-seconds 3690 ...
```

## Security note

`youtube-cookies.txt` can sign in as you. Do not commit it to git or share it.
Keep it only under `logs\` (local).
