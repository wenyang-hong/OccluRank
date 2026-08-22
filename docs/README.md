# OccluRank project page

This directory contains the static GitHub Pages website for OccluRank.

## Publish with GitHub Pages

1. Commit the `docs/` directory to the OccluRank repository.
2. Open **Settings > Pages** in GitHub.
3. Select **Deploy from a branch**.
4. Choose the `main` branch and the `/docs` folder.
5. Save the configuration.

The expected project-page URL is:

```text
https://wenyang-hong.github.io/OccluRank/
```

## Add release links later

The Paper, Dataset, and Benchmark buttons in `index.html` are currently disabled and marked `Coming soon`.

After a resource is released, replace the corresponding `<span>` with an anchor. For example:

```html
<a class="button" href="https://example.com/occlulayout">Dataset</a>
```

The same links appear near the top of the repository `README.md`. They currently point to internal release sections. Replace those anchor targets with the public URLs when available.

## Local check

From the repository root, run:

```bash
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000/`.

