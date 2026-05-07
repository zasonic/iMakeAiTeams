import { describe, expect, it } from "vitest";

import { t } from "./index";

describe("t()", () => {
  it("maps router_log to its user-facing label", () => {
    expect(t("router_log")).toBe("Smart Routing History");
  });

  it("returns the input unchanged when no mapping exists", () => {
    expect(t("not_a_real_key")).toBe("not_a_real_key");
  });
});
