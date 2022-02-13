.PHONY: book
book:
		jupyter-book build .
		
.PHONY: site
site:
		ghp-import -n -p -f _build/html
