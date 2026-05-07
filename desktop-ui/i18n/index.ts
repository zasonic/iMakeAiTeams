import en from "./en.json";

const dictionary = en as Record<string, string>;

export function t(key: string): string {
  return dictionary[key] ?? key;
}
