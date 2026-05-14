import { checkSessionStatus, parseResume } from "./api";

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

describe("parseResume", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("posts FormData and returns parsed payload", async () => {
    const payload = {
      extracted_text: "Jane Doe",
      skills: ["python", "react"],
      summary: "Jane Doe summary",
      page_count: 1,
      truncated: false,
    };
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => payload,
    });

    const file = new File(["%PDF-1.4"], "resume.pdf", {
      type: "application/pdf",
    });
    const result = await parseResume(file);

    expect(result).toEqual(payload);
    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [, init] = global.fetch.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect(init.body.get("file")).toBe(file);
  });

  test("throws with backend error detail on non-ok response", async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 415,
      json: async () => ({ detail: "Only PDF resumes are supported" }),
    });

    const file = new File(["not a pdf"], "resume.txt", { type: "text/plain" });
    await expect(parseResume(file)).rejects.toThrow(
      /Only PDF resumes are supported/,
    );
  });
});
