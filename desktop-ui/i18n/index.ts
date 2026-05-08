import en from "./en.json";

const dict = en as Record<string, string>;

export function t(key: string): string {
  return dict[key] ?? key;
}
