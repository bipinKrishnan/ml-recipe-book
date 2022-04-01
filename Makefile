.PHONY: book
book:
		jupyter-book build .

.PHONY: new-book
new-book:
		rm -r _build && jupyter-book build .
		
.PHONY: site
site:
		ghp-import -n -p -f _build/html
