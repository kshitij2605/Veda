import { type ReactNode } from 'react'
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from '@tanstack/react-table'
import { cn } from '../../lib/cn'

export interface DataTableProps<T> {
  columns: ColumnDef<T>[]
  data: T[]
  onRowClick?: (row: T) => void
  empty?: ReactNode
}

/** Generic token-styled table over TanStack. (Virtualization arrives in a
 *  later phase when a page actually needs >50 rows.) */
export function DataTable<T>({ columns, data, onRowClick, empty }: DataTableProps<T>) {
  const table = useReactTable({ data, columns, getCoreRowModel: getCoreRowModel() })

  if (data.length === 0 && empty) return <>{empty}</>

  return (
    <table className="w-full text-[12px]">
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id} className="text-left text-[var(--color-text-subtle)]">
            {hg.headers.map((h) => (
              <th key={h.id} className="px-4 py-2 font-medium">
                {h.isPlaceholder ? null : flexRender(h.column.columnDef.header, h.getContext())}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr
            key={row.id}
            onClick={() => onRowClick?.(row.original)}
            className={cn(
              'border-t border-[var(--color-border)] transition-colors',
              onRowClick && 'cursor-pointer hover:bg-[var(--color-bg-hover)]',
            )}
          >
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id} className="px-4 py-2.5 text-[var(--color-text)]">
                {cell.column.columnDef.cell
                  ? flexRender(cell.column.columnDef.cell, cell.getContext())
                  : String(cell.getValue() ?? '')}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
