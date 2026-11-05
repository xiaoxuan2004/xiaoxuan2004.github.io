# Xuan Xiao — Academic Homepage

Four sections: About, Education & Experience, Research, and Beyond Research.
The homepage uses academic.css and academic.js. The PIC course project remains available from the footer.

## Local preview

Run `python -m http.server 8000 --bind 127.0.0.1 --directory website` from the project root.
Open http://127.0.0.1:8000/ . Publish only the contents of website/.

## Edit content and photos

Homepage text: `site_source/academic_homepage.template.html`.
Styles and interactions: `site_source/academic.css` and `site_source/academic.js`.
Run `python site_source/build_academic_homepage.py` to regenerate only the homepage.
The complete `python site_source/build_portfolio.py` build also ends with the academic homepage.

Place actual photos in `website/assets/photos/`. In `profile.json`, set:

```json
"homepage": {
  "email": "2611210700@pku.edu.cn",
  "portrait": "assets/photos/portrait.jpg",
  "gallery": [
    {"src": "assets/photos/life-01.jpg", "alt": "A descriptive photo caption", "caption": "A moment beyond the lab"}
  ]
}
```

Until real photos are supplied, the page shows explicit placeholders. No stock portrait is used.
Do not add nonexistent file paths. Contact information and the address are shown once in the profile sidebar.
