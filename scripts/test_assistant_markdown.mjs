import assert from 'node:assert/strict'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function render(content) {
  return renderToStaticMarkup(
    createElement(ReactMarkdown, { remarkPlugins: [remarkGfm], skipHtml: true }, content),
  )
}

const content = `**Titre de l’activité**

### Vocabulaire cible

| Mot (arabe) | Français |
| --- | --- |
| أمي | Ma mère |

&#x20;

<script>window.__unsafe = true</script>`
const html = render(content)

assert.match(html, /<strong>Titre de l’activité<\/strong>/)
assert.match(html, /<h3>Vocabulaire cible<\/h3>/)
assert.match(html, /<table>/)
assert.match(html, /أمي/)
assert.doesNotMatch(html, /&amp;#x20;/)
assert.doesNotMatch(html, /<script>/)
assert.doesNotMatch(html, /window\.__unsafe/)

console.log('Assistant Markdown rendering tests passed.')
