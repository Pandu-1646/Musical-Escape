$port = 8181
$prefix = "http://localhost:$port/"
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add($prefix)

try {
    $listener.Start()
    Write-Output "[+] Local Web Server is LIVE at: $prefix"
} catch {
    Write-Error "[!] Failed to start HttpListener: $_"
    exit 1
}

$dir = "C:\Users\bhima\.gemini\antigravity\scratch\genre_classifier"

while ($listener.IsListening) {
    try {
        $context = $listener.GetContext()
        $req = $context.Request
        $res = $context.Response

        $path = $req.Url.LocalPath.TrimStart('/')
        if ([string]::IsNullOrWhiteSpace($path)) {
            $path = "index.html"
        }
        $path = $path.Replace('/', '\')
        $fullPath = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($dir, $path))

        if ($fullPath.StartsWith($dir, [System.StringComparison]::OrdinalIgnoreCase) -and [System.IO.File]::Exists($fullPath)) {
            $bytes = [System.IO.File]::ReadAllBytes($fullPath)
            $ext = [System.IO.Path]::GetExtension($fullPath).ToLower()
            
            if ($ext -eq ".html" -or $ext -eq ".htm") { $res.ContentType = "text/html; charset=utf-8" }
            elseif ($ext -eq ".jpg" -or $ext -eq ".jpeg") { $res.ContentType = "image/jpeg" }
            elseif ($ext -eq ".png") { $res.ContentType = "image/png" }
            elseif ($ext -eq ".js") { $res.ContentType = "application/javascript; charset=utf-8" }
            elseif ($ext -eq ".css") { $res.ContentType = "text/css; charset=utf-8" }
            elseif ($ext -eq ".json") { $res.ContentType = "application/json; charset=utf-8" }
            elseif ($ext -eq ".mp3") { $res.ContentType = "audio/mpeg" }
            elseif ($ext -eq ".wav") { $res.ContentType = "audio/wav" }
            else { $res.ContentType = "application/octet-stream" }

            $res.ContentLength64 = $bytes.Length
            $res.Headers.Add("Access-Control-Allow-Origin", "*")
            $res.Headers.Add("Cache-Control", "no-cache, no-store, must-revalidate")
            $res.StatusCode = 200

            if ($req.HttpMethod -ne "HEAD") {
                $res.OutputStream.Write($bytes, 0, $bytes.Length)
            }
        } else {
            $res.StatusCode = 404
            $err = [System.Text.Encoding]::UTF8.GetBytes("404 Not Found")
            $res.OutputStream.Write($err, 0, $err.Length)
        }
        $res.OutputStream.Close()
    } catch {
        # continue loop
    }
}
