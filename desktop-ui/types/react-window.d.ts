// Minimal ambient types for react-window. The package ships Flow types but
// no TypeScript .d.ts and we don't want to pull in @types/react-window just
// to virtualize one list. Only the surface ChatView consumes is declared.

declare module "react-window" {
  import type { CSSProperties, Component, ComponentType } from "react";

  export interface ListChildComponentProps<TData = unknown> {
    index: number;
    style: CSSProperties;
    data: TData;
    isScrolling?: boolean;
  }

  export interface VariableSizeListProps<TData = unknown> {
    children: ComponentType<ListChildComponentProps<TData>>;
    height: number;
    width: number | string;
    itemCount: number;
    itemSize: (index: number) => number;
    itemData?: TData;
    itemKey?: (index: number, data: TData) => string | number;
    estimatedItemSize?: number;
    overscanCount?: number;
    className?: string;
    style?: CSSProperties;
  }

  export class VariableSizeList<TData = unknown> extends Component<VariableSizeListProps<TData>> {
    scrollToItem(index: number, align?: "auto" | "smart" | "center" | "end" | "start"): void;
    resetAfterIndex(index: number, shouldForceUpdate?: boolean): void;
  }
}
