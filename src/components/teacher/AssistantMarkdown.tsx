import type { ComponentPropsWithoutRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type MarkdownProps = { content: string }

type ElementProps<T extends keyof JSX.IntrinsicElements> = ComponentPropsWithoutRef<T>

/** Renders trusted text as Markdown while deliberately ignoring model-supplied HTML. */
export function AssistantMarkdown({ content }: MarkdownProps) {
  return <div dir="auto" className="break-words text-sm leading-6 text-slate-700">
    <ReactMarkdown
      skipHtml
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }: ElementProps<'h1'>) => <h3 className="mb-3 mt-5 text-lg font-bold text-slate-900 first:mt-0">{children}</h3>,
        h2: ({ children }: ElementProps<'h2'>) => <h3 className="mb-3 mt-5 text-base font-bold text-slate-900 first:mt-0">{children}</h3>,
        h3: ({ children }: ElementProps<'h3'>) => <h4 className="mb-2 mt-4 text-sm font-bold text-slate-900 first:mt-0">{children}</h4>,
        p: ({ children }: ElementProps<'p'>) => <p className="mb-3 last:mb-0">{children}</p>,
        ul: ({ children }: ElementProps<'ul'>) => <ul className="mb-3 list-disc space-y-1 ps-5 last:mb-0">{children}</ul>,
        ol: ({ children }: ElementProps<'ol'>) => <ol className="mb-3 list-decimal space-y-1 ps-5 last:mb-0">{children}</ol>,
        strong: ({ children }: ElementProps<'strong'>) => <strong className="font-bold text-slate-900">{children}</strong>,
        table: ({ children }: ElementProps<'table'>) => <div className="mb-3 overflow-x-auto rounded-lg border border-slate-200 last:mb-0"><table className="min-w-full border-collapse text-start text-xs">{children}</table></div>,
        th: ({ children }: ElementProps<'th'>) => <th className="border-b border-slate-200 bg-slate-50 px-3 py-2 font-semibold text-slate-800">{children}</th>,
        td: ({ children }: ElementProps<'td'>) => <td className="border-b border-slate-100 px-3 py-2 align-top last:border-b-0">{children}</td>,
      }}
    >
      {content}
    </ReactMarkdown>
  </div>
}
