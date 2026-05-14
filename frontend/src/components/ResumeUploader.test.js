import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ResumeUploader from "./ResumeUploader";

function makePdfFile(name = "resume.pdf", size = 1024) {
  const blob = new Blob([new Uint8Array(size)], { type: "application/pdf" });
  return new File([blob], name, { type: "application/pdf" });
}

describe("ResumeUploader", () => {
  test("renders empty state with upload button when no resumeData", () => {
    render(
      <ResumeUploader
        resumeData={null}
        onUpload={jest.fn()}
        onClear={jest.fn()}
        onRemoveSkill={jest.fn()}
        disabled={false}
      />
    );
    expect(screen.getByRole("button", { name: /upload resume/i })).toBeInTheDocument();
    expect(screen.queryByTestId("resume-skill-chips")).not.toBeInTheDocument();
  });

  test("calls onUpload when a PDF file is selected", async () => {
    const onUpload = jest.fn().mockResolvedValue(undefined);
    render(
      <ResumeUploader
        resumeData={null}
        onUpload={onUpload}
        onClear={jest.fn()}
        onRemoveSkill={jest.fn()}
        disabled={false}
      />
    );

    const file = makePdfFile();
    const input = screen.getByTestId("resume-file-input");
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(onUpload).toHaveBeenCalledWith(file);
    });
  });

  test("rejects non-PDF files without calling onUpload", async () => {
    const onUpload = jest.fn();
    render(
      <ResumeUploader
        resumeData={null}
        onUpload={onUpload}
        onClear={jest.fn()}
        onRemoveSkill={jest.fn()}
        disabled={false}
      />
    );

    const txt = new File(["plain text"], "resume.txt", { type: "text/plain" });
    const input = screen.getByTestId("resume-file-input");
    fireEvent.change(input, { target: { files: [txt] } });

    expect(onUpload).not.toHaveBeenCalled();
    expect(await screen.findByRole("alert")).toHaveTextContent(/PDF/);
  });

  test("renders loaded state with skill chips when resumeData is set", () => {
    render(
      <ResumeUploader
        resumeData={{
          fileName: "resume.pdf",
          skills: ["python", "react", "aws"],
          summary: "...",
          pageCount: 1,
          truncated: false,
        }}
        onUpload={jest.fn()}
        onClear={jest.fn()}
        onRemoveSkill={jest.fn()}
        disabled={false}
      />
    );

    expect(screen.getByText(/resume\.pdf/)).toBeInTheDocument();
    const chips = screen.getByTestId("resume-skill-chips");
    expect(chips).toHaveTextContent("python");
    expect(chips).toHaveTextContent("react");
    expect(chips).toHaveTextContent("aws");
  });

  test("clicking a skill chip calls onRemoveSkill with the skill", () => {
    const onRemoveSkill = jest.fn();
    render(
      <ResumeUploader
        resumeData={{
          fileName: "resume.pdf",
          skills: ["python", "react"],
          summary: "...",
          pageCount: 1,
          truncated: false,
        }}
        onUpload={jest.fn()}
        onClear={jest.fn()}
        onRemoveSkill={onRemoveSkill}
        disabled={false}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /python/i }));
    expect(onRemoveSkill).toHaveBeenCalledWith("python");
  });

  test("clicking the clear button calls onClear", () => {
    const onClear = jest.fn();
    render(
      <ResumeUploader
        resumeData={{
          fileName: "resume.pdf",
          skills: ["python"],
          summary: "...",
          pageCount: 1,
          truncated: false,
        }}
        onUpload={jest.fn()}
        onClear={onClear}
        onRemoveSkill={jest.fn()}
        disabled={false}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /remove resume/i }));
    expect(onClear).toHaveBeenCalled();
  });

  test("surfaces upload errors as alerts", async () => {
    const onUpload = jest.fn().mockRejectedValue(new Error("Could not read PDF"));
    render(
      <ResumeUploader
        resumeData={null}
        onUpload={onUpload}
        onClear={jest.fn()}
        onRemoveSkill={jest.fn()}
        disabled={false}
      />
    );

    fireEvent.change(screen.getByTestId("resume-file-input"), {
      target: { files: [makePdfFile()] },
    });

    expect(await screen.findByRole("alert")).toHaveTextContent(/Could not read PDF/);
  });

  test("disables interactions when disabled prop is true", () => {
    render(
      <ResumeUploader
        resumeData={null}
        onUpload={jest.fn()}
        onClear={jest.fn()}
        onRemoveSkill={jest.fn()}
        disabled={true}
      />
    );
    expect(screen.getByRole("button", { name: /upload resume/i })).toBeDisabled();
  });
});
