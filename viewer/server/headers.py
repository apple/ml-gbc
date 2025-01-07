headers = [
    (
        "accept",
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    ),
    ("accept-encoding", "gzip, deflate, br"),
    (
        "accept-language",
        "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,af;q=0.6,ja;q=0.5,zh-CN;q=0.4",
    ),
    ("cache-control", "max-age=0"),
    ("if-modified-since", "Mon, 04 May 2020 12:41:48 GMT"),
    ("referer", "https://www.pixiv.net/artworks/81295155"),
    ("sec-fetch-dest", "document"),
    ("sec-fetch-mode", "navigate"),
    ("sec-fetch-site", "none"),
    ("sec-fetch-user", "?1"),
    ("upgrade-insecure-requests", "1"),
    (
        "user-agent",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/81.0.4044.129 Safari/537.36",
    ),
]

headers_dict = {i.strip(): j.strip() for (i, j) in headers}
