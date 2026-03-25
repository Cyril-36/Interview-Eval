const ROLE_ALIASES = {
  swe: "Software Engineer",
  "software engineer": "Software Engineer",
  "data scientist": "Data Scientist",
  "ml engineer": "ML Engineer",
  "machine learning engineer": "ML Engineer",
  "full stack developer": "Full Stack Developer",
  "full-stack developer": "Full Stack Developer",
  "product manager": "Product Manager",
  "frontend engineer": "Frontend Engineer",
  "backend engineer": "Backend Engineer",
  "data analyst": "Data Analyst",
  "devops engineer": "DevOps Engineer",
};

const ROLE_KEYWORDS = [
  "engineer",
  "developer",
  "scientist",
  "manager",
  "analyst",
  "designer",
  "researcher",
  "architect",
  "consultant",
  "administrator",
];

const LEVEL_ALIASES = {
  intern: "Intern",
  internship: "Intern",
  junior: "Junior",
  jr: "Junior",
  "mid-level": "Mid-Level",
  "mid level": "Mid-Level",
  midlevel: "Mid-Level",
  senior: "Senior",
  lead: "Lead",
};

const DIFFICULTY_ALIASES = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
};

const CATEGORY_ALIASES = {
  "data structures": "Data Structures",
  "data structure": "Data Structures",
  algorithms: "Algorithms",
  algorithm: "Algorithms",
  "system design": "System Design",
  systems: "System Design",
  oop: "OOP",
  oops: "OOP",
  "object oriented programming": "OOP",
  "object-oriented programming": "OOP",
  database: "Databases",
  databases: "Databases",
  db: "Databases",
  "machine learning": "Machine Learning",
  ml: "Machine Learning",
  behavioral: "Behavioral",
  behaviour: "Behavioral",
  behaviourial: "Behavioral",
  behavioural: "Behavioral",
  "behavioral question": "Behavioral",
  "behavioral questions": "Behavioral",
  "behavioural question": "Behavioral",
  "behavioural questions": "Behavioral",
  "web development": "Web Development",
  "web dev": "Web Development",
};

function toTitleCase(value) {
  const uppercaseWords = new Map([
    ["ai", "AI"],
    ["ml", "ML"],
    ["nlp", "NLP"],
    ["qa", "QA"],
    ["ui", "UI"],
    ["ux", "UX"],
    ["devops", "DevOps"],
  ]);

  return value
    .split(/\s+/)
    .map((part) => {
      const lower = part.toLowerCase();
      if (uppercaseWords.has(lower)) {
        return uppercaseWords.get(lower);
      }
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(" ");
}

function normalizeSetupToken(value) {
  return value.replace(/[.,!?;:]+$/g, "").trim();
}

export function parseSetupValue(stepKey, rawValue) {
  const trimmed = rawValue.trim().replace(/\s+/g, " ");
  const cleaned = normalizeSetupToken(trimmed);

  if (stepKey === "role") {
    const exactMatch = ROLE_ALIASES[cleaned.toLowerCase()];
    if (exactMatch) {
      return { value: exactMatch };
    }

    if (!/^[A-Za-z][A-Za-z\s/-]{1,48}[A-Za-z]$/.test(cleaned)) {
      return {
        error: "Please enter a valid role, e.g. Software Engineer or Data Scientist.",
      };
    }

    const lower = cleaned.toLowerCase();
    if (!ROLE_KEYWORDS.some((keyword) => lower.includes(keyword))) {
      return {
        error: "Please enter a role like Software Engineer, Data Scientist, or Product Manager.",
      };
    }

    return { value: toTitleCase(cleaned) };
  }

  if (stepKey === "num_questions") {
    const num = parseInt(cleaned, 10);
    if (isNaN(num) || num < 1 || num > 10) {
      return { error: "Please enter a number between 1 and 10." };
    }
    return { value: String(num), numericValue: num };
  }

  if (stepKey === "level") {
    const normalized = LEVEL_ALIASES[cleaned.toLowerCase()];
    if (!normalized) {
      return {
        error: "Please enter one of: Intern, Junior, Mid-Level, Senior, Lead.",
      };
    }
    return { value: normalized };
  }

  if (stepKey === "difficulty") {
    const normalized = DIFFICULTY_ALIASES[cleaned.toLowerCase()];
    if (!normalized) {
      return { error: "Please enter one of: Easy, Medium, Hard." };
    }
    return { value: normalized };
  }

  if (stepKey === "category") {
    const normalized = CATEGORY_ALIASES[cleaned.toLowerCase()];
    if (!normalized) {
      return {
        error: "Please enter one of: Data Structures, Algorithms, System Design, OOP, Databases, Machine Learning, Behavioral, Web Development.",
      };
    }
    return { value: normalized };
  }

  return { value: trimmed };
}
