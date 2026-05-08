import { describe, expect, it } from "vitest";

import { t } from "./index";

describe("t", () => {
  it("returns mapped string for known key", () => {
    expect(t("chat_orchestrator")).toBe("Conversation Engine");
  });

  it("returns input unchanged for unknown key", () => {
    expect(t("not_a_real_key")).toBe("not_a_real_key");
  });

  it("returns input unchanged for a trimmed-out internal name", () => {
    expect(t("router_log")).toBe("router_log");
  });
});
