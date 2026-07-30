import { cn } from '../../utils/cn'

export function Section({ children, className, id, ...props }) {
  return (
    <section id={id} className={cn('py-10 sm:py-14 lg:py-20', className)} {...props}>
      {children}
    </section>
  )
}
