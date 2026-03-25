import { parseSetupValue } from "./setup";

describe("parseSetupValue", () => {
  test("normalizes a known role alias", () => {
    expect(parseSetupValue("role", "software engineer")).toEqual({
      value: "Software Engineer",
    });
  });

  test("rejects a nonsense role", () => {
    expect(parseSetupValue("role", "asdfghjkl")).toEqual({
      error: "Please enter a role like Software Engineer, Data Scientist, or Product Manager.",
    });
  });

  test("normalizes a known category alias", () => {
    expect(parseSetupValue("category", "behavioural questions")).toEqual({
      value: "Behavioral",
    });
  });

  test("accepts trailing punctuation on difficulty inputs", () => {
    expect(parseSetupValue("difficulty", "easy,")).toEqual({
      value: "Easy",
    });
  });

  test("accepts trailing punctuation on level inputs", () => {
    expect(parseSetupValue("level", "intern.")).toEqual({
      value: "Intern",
    });
  });

  test("rejects an unsupported category", () => {
    expect(parseSetupValue("category", "random stuff")).toEqual({
      error: "Please enter one of: Data Structures, Algorithms, System Design, OOP, Databases, Machine Learning, Behavioral, Web Development.",
    });
  });
});
