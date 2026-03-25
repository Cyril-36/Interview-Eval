import { checkSessionStatus } from "./api";

describe("checkSessionStatus", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("returns unavailable when the fetch rejects", async () => {
    global.fetch = jest.fn().mockRejectedValue(new Error("network down"));

    await expect(checkSessionStatus("sess-123")).resolves.toEqual({
      kind: "unavailable",
    });
  });

  test("returns missing on 404", async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 404,
    });

    await expect(checkSessionStatus("sess-123")).resolves.toEqual({
      kind: "missing",
    });
  });
});
