"""
Synthetic Dataset Generator — AI Semantic Matching Engine
===========================================================
Generates two linked, semantically-coherent datasets:
    1. students.csv           -> student/mentee profiles
    2. businesses.csv         -> business/mentor profiles
    3. ground_truth_matches.csv -> known good student<->business pairs (for evaluation)
    4. feedback_logs.csv      -> simulated post-match feedback (for the future feedback loop)

Design goal: attributes within a profile are NOT random-independent — a student's
major drives their skills/interests/career goals, and a business's industry drives
its mentor's expertise/offerings. This makes the dataset usable for realistically
testing semantic similarity + embedding quality, rather than just schema-shape testing.
"""

import csv
import random
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# 1. DOMAIN KNOWLEDGE MAPS (keeps generated profiles internally coherent)
# ---------------------------------------------------------------------------

DOMAINS = {
    "Technology / Software Engineering": {
        "majors": ["Computer Science", "Software Engineering", "Information Technology", "Data Science"],
        "skills": ["Python", "JavaScript", "Cloud Computing (AWS)", "Machine Learning", "SQL",
                   "React", "System Design", "DevOps", "Cybersecurity Basics", "Git/Version Control"],
        "interests": ["building side projects", "open-source contribution", "hackathons",
                      "AI research", "competitive programming"],
        "roles": ["Senior Software Engineer", "Engineering Manager", "CTO", "Product Engineer",
                  "Data Science Lead", "DevOps Architect"],
        "company_examples": ["Nimbus Cloud Systems", "Vertex Software Labs", "Coreline Technologies",
                              "PixelForge Studios", "Datastream Analytics"],
    },
    "Finance / Investment": {
        "majors": ["Finance", "Economics", "Accounting", "Business Administration"],
        "skills": ["Financial Modeling", "Excel/VBA", "Valuation", "Risk Analysis",
                   "Bloomberg Terminal", "Bookkeeping", "Investment Research", "Regulatory Compliance"],
        "interests": ["stock market analysis", "personal finance", "case competitions",
                      "startup investing", "economic policy"],
        "roles": ["Investment Analyst", "Portfolio Manager", "Finance Director", "CFO",
                  "Risk Management Lead", "Wealth Advisor"],
        "company_examples": ["Meridian Capital Partners", "Blackridge Financial Group",
                              "Harbor & Vine Advisors", "Northgate Investments", "Clearpoint Wealth"],
    },
    "Healthcare / Life Sciences": {
        "majors": ["Biology", "Nursing", "Public Health", "Biomedical Engineering", "Pre-Med"],
        "skills": ["Clinical Research", "Patient Care", "Medical Data Analysis", "Lab Techniques",
                   "Healthcare Compliance (HIPAA)", "Biostatistics", "EHR Systems"],
        "interests": ["community health outreach", "medical research", "volunteering at clinics",
                      "health tech innovation"],
        "roles": ["Clinical Research Director", "Chief Medical Officer", "Healthcare Consultant",
                  "Hospital Administrator", "Biotech Program Manager"],
        "company_examples": ["Wellspring Health Group", "Cascade Medical Partners",
                              "BioNova Therapeutics", "Ridgeline Clinical Research", "Vitalis Health"],
    },
    "Marketing / Communications": {
        "majors": ["Marketing", "Communications", "Journalism", "Public Relations"],
        "skills": ["SEO/SEM", "Content Strategy", "Social Media Management", "Brand Strategy",
                   "Copywriting", "Google Analytics", "Campaign Management", "Public Speaking"],
        "interests": ["content creation", "brand storytelling", "social media trends",
                      "influencer marketing", "market research"],
        "roles": ["Marketing Director", "Brand Strategist", "Communications Lead",
                  "Chief Marketing Officer", "Growth Marketing Manager"],
        "company_examples": ["Beacon Brand Studio", "Momentum Media Group", "Clearvoice Communications",
                              "Northstar Marketing Collective", "Ember Creative Agency"],
    },
    "Design / Creative": {
        "majors": ["Graphic Design", "UX/UI Design", "Fine Arts", "Industrial Design"],
        "skills": ["Figma", "Adobe Creative Suite", "User Research", "Prototyping",
                   "Typography", "Design Systems", "3D Modeling", "Motion Graphics"],
        "interests": ["visual storytelling", "design portfolios", "art exhibitions",
                      "product design critique", "sketching"],
        "roles": ["Creative Director", "Head of Product Design", "UX Research Lead",
                  "Design Studio Founder", "Senior Visual Designer"],
        "company_examples": ["Studio Lumen", "Formwork Design Collective", "Aperture Creative House",
                              "Northline Design Co.", "Palette & Grid Studio"],
    },
    "Consulting / Strategy": {
        "majors": ["Business Administration", "Economics", "Management", "International Relations"],
        "skills": ["Strategic Planning", "Data-Driven Decision Making", "Client Management",
                   "Market Analysis", "Process Optimization", "Presentation Design", "Negotiation"],
        "interests": ["case study competitions", "business strategy", "consulting club",
                      "cross-industry research"],
        "roles": ["Management Consultant", "Strategy Director", "Principal Consultant",
                  "Operations Partner", "Business Transformation Lead"],
        "company_examples": ["Ashford Strategy Group", "Keystone Consulting Partners",
                              "Bridgeway Advisory", "Lattice Business Solutions", "Vantage Point Consulting"],
    },
    "Education / Nonprofit": {
        "majors": ["Education", "Social Work", "Sociology", "Public Policy"],
        "skills": ["Curriculum Design", "Community Outreach", "Grant Writing", "Program Management",
                   "Public Speaking", "Policy Research", "Fundraising"],
        "interests": ["mentorship programs", "education equity", "volunteering",
                      "youth advocacy", "policy reform"],
        "roles": ["Program Director", "Nonprofit Executive Director", "Education Policy Advisor",
                  "Community Impact Lead", "Grants Manager"],
        "company_examples": ["Rise Together Foundation", "Bright Path Education Initiative",
                              "Commonwealth Youth Alliance", "Openhand Community Fund", "Elevate Education Corp"],
    },
    "Media / Entertainment": {
        "majors": ["Film Studies", "Media Production", "Communications", "Music Business"],
        "skills": ["Video Editing", "Scriptwriting", "Production Management", "Sound Design",
                   "Social Media Content", "Storyboarding", "Talent Coordination"],
        "interests": ["filmmaking", "podcasting", "music production", "streaming platforms",
                      "creative writing"],
        "roles": ["Executive Producer", "Studio Head", "Content Strategy Director",
                  "Talent Manager", "Head of Production"],
        "company_examples": ["Lightframe Studios", "Wavecrest Media Group", "Nightowl Productions",
                              "Fable & Frame Entertainment", "Reelhouse Studios"],
    },
}

CITIES = ["Austin, TX", "Seattle, WA", "New York, NY", "Chicago, IL", "San Francisco, CA",
          "Boston, MA", "Denver, CO", "Atlanta, GA", "Toronto, ON", "London, UK",
          "Bengaluru, IN", "Coimbatore, IN", "Berlin, DE", "Singapore, SG"]

UNIVERSITIES = ["Riverdale State University", "Ashwood Institute of Technology", "Northfield University",
                "Grantham College", "Cedar Valley University", "Milbrook Institute",
                "Lakeshore University", "Prescott State College", "Hawthorne University"]

YEARS_OF_STUDY = ["1st Year", "2nd Year", "3rd Year", "Final Year", "Graduate Student"]
MENTORSHIP_TYPES = ["Career Guidance", "Technical Skill-Building", "Networking & Industry Exposure",
                     "Resume & Interview Prep", "Entrepreneurship Guidance", "Leadership Development"]
MENTORING_STYLES = ["structured and goal-oriented", "casual and conversational",
                     "hands-on and project-based", "advisory and big-picture",
                     "high-frequency check-ins", "flexible, on-demand"]
COMPANY_SIZES = ["Startup (1-50 employees)", "Mid-size (51-500 employees)",
                  "Large Enterprise (500+ employees)"]

domain_keys = list(DOMAINS.keys())

# ---------------------------------------------------------------------------
# 2. STUDENT PROFILE GENERATOR
# ---------------------------------------------------------------------------

def generate_student(student_id):
    domain_key = random.choice(domain_keys)
    d = DOMAINS[domain_key]

    name = fake.name()
    major = random.choice(d["majors"])
    year = random.choice(YEARS_OF_STUDY)
    university = random.choice(UNIVERSITIES)
    location = random.choice(CITIES)
    gpa = round(random.uniform(2.8, 4.0), 2)

    # 3-5 skills, mostly domain-relevant with 1 cross-domain wildcard for realism
    skills = random.sample(d["skills"], k=min(4, len(d["skills"])))
    if random.random() < 0.3:
        other_domain = random.choice([k for k in domain_keys if k != domain_key])
        skills.append(random.choice(DOMAINS[other_domain]["skills"]))

    interests = random.sample(d["interests"], k=min(3, len(d["interests"])))
    preferred_mentorship = random.sample(MENTORSHIP_TYPES, k=random.choice([1, 2]))
    availability_hrs_per_week = random.choice([1, 2, 3, 4, 5])
    target_role = random.choice(d["roles"])

    bio = (
        f"{name} is a {year.lower()} student majoring in {major} at {university}, based in {location}. "
        f"Passionate about {', '.join(interests)}, with hands-on skills in {', '.join(skills)}. "
        f"Aspiring to grow into a role such as {target_role} within the {domain_key.split(' / ')[0]} field. "
        f"Currently seeking mentorship focused on {', '.join(preferred_mentorship).lower()}, "
        f"and available for approximately {availability_hrs_per_week} hours per week."
    )

    return {
        "student_id": f"STU{student_id:04d}",
        "name": name,
        "age": random.randint(18, 27),
        "university": university,
        "major": major,
        "year_of_study": year,
        "gpa": gpa,
        "location": location,
        "domain_category": domain_key,
        "skills": "; ".join(skills),
        "interests": "; ".join(interests),
        "target_role": target_role,
        "preferred_mentorship_type": "; ".join(preferred_mentorship),
        "availability_hrs_per_week": availability_hrs_per_week,
        "semantic_profile_text": bio,
    }


# ---------------------------------------------------------------------------
# 3. BUSINESS / MENTOR PROFILE GENERATOR
# ---------------------------------------------------------------------------

def generate_business(business_id):
    domain_key = random.choice(domain_keys)
    d = DOMAINS[domain_key]

    company_name = random.choice(d["company_examples"]) + random.choice(["", " Inc.", " Group", ""])
    mentor_name = fake.name()
    title = random.choice(d["roles"])
    years_experience = random.randint(5, 25)
    company_size = random.choice(COMPANY_SIZES)
    location = random.choice(CITIES)
    mentoring_style = random.choice(MENTORING_STYLES)

    expertise = random.sample(d["skills"], k=min(4, len(d["skills"])))
    offerings = random.sample(MENTORSHIP_TYPES, k=random.choice([1, 2]))

    bio = (
        f"{mentor_name} is a {title} at {company_name}, a {company_size.lower()} organization in the "
        f"{domain_key} space, based in {location}. With {years_experience} years of experience and deep "
        f"expertise in {', '.join(expertise)}, {mentor_name.split()[0]} mentors in a {mentoring_style} style. "
        f"Offers guidance in {', '.join(offerings).lower()}, and is looking to support motivated students "
        f"interested in breaking into {domain_key.split(' / ')[0]}."
    )

    return {
        "business_id": f"BUS{business_id:04d}",
        "company_name": company_name,
        "mentor_name": mentor_name,
        "title": title,
        "domain_category": domain_key,
        "years_experience": years_experience,
        "company_size": company_size,
        "location": location,
        "expertise_areas": "; ".join(expertise),
        "mentoring_style": mentoring_style,
        "offerings": "; ".join(offerings),
        "semantic_profile_text": bio,
    }


# ---------------------------------------------------------------------------
# 4. GROUND TRUTH MATCHES (same domain_category = a "good" match, for eval)
# ---------------------------------------------------------------------------

def generate_ground_truth(students, businesses, top_k=3):
    rows = []
    by_domain_biz = {}
    for b in businesses:
        by_domain_biz.setdefault(b["domain_category"], []).append(b)

    for s in students:
        candidates = by_domain_biz.get(s["domain_category"], [])
        if not candidates:
            continue
        matches = random.sample(candidates, k=min(top_k, len(candidates)))
        for rank, m in enumerate(matches, start=1):
            rows.append({
                "student_id": s["student_id"],
                "business_id": m["business_id"],
                "match_rank": rank,
                "match_reason": f"Shared domain: {s['domain_category']}",
            })
    return rows


# ---------------------------------------------------------------------------
# 5. SIMULATED FEEDBACK LOGS (for the future feedback-loop feature)
# ---------------------------------------------------------------------------

def generate_feedback(ground_truth_rows):
    rows = []
    for r in ground_truth_rows:
        if random.random() < 0.7:  # not every match gets feedback
            rating = random.choices([1, 2, 3, 4, 5], weights=[3, 5, 12, 35, 45])[0]
            accepted = rating >= 3
            rows.append({
                "student_id": r["student_id"],
                "business_id": r["business_id"],
                "rating_1_to_5": rating,
                "accepted_match": accepted,
                "feedback_comment": random.choice([
                    "Great alignment on skills and goals.",
                    "Mentor's experience was very relevant.",
                    "Good match, would recommend.",
                    "Somewhat helpful but not a strong fit.",
                    "Not enough overlap in interests.",
                    "Excellent guidance on career direction.",
                    "Scheduling was hard, but content was valuable.",
                ]),
            })
    return rows


# ---------------------------------------------------------------------------
# 6. BUILD + WRITE CSVs
# ---------------------------------------------------------------------------

N_STUDENTS = 500
N_BUSINESSES = 200

students = [generate_student(i) for i in range(1, N_STUDENTS + 1)]
businesses = [generate_business(i) for i in range(1, N_BUSINESSES + 1)]
ground_truth = generate_ground_truth(students, businesses, top_k=3)
feedback = generate_feedback(ground_truth)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


write_csv("/home/claude/dataset_gen/students.csv", students, list(students[0].keys()))
write_csv("/home/claude/dataset_gen/businesses.csv", businesses, list(businesses[0].keys()))
write_csv("/home/claude/dataset_gen/ground_truth_matches.csv", ground_truth, list(ground_truth[0].keys()))
write_csv("/home/claude/dataset_gen/feedback_logs.csv", feedback, list(feedback[0].keys()))

print(f"Students: {len(students)}")
print(f"Businesses: {len(businesses)}")
print(f"Ground truth match pairs: {len(ground_truth)}")
print(f"Feedback logs: {len(feedback)}")
