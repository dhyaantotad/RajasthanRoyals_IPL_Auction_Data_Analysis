import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def load_one(prefix):
    found = sorted(ROOT.glob(f"{prefix}*.csv"))
    if not found:
        raise FileNotFoundError(f"Missing CSV beginning with {prefix}")
    return pd.read_csv(found[0])

ball = load_one("ball_by_ball")
matches = load_one("matches")
competitions = load_one("competitions")

IPL_IDS = [1358, 1715, 2193, 2725, 3298]
BAT_MIN_PRESSURE_BALLS, BAT_MIN_PRESSURE_MATCHES, BAT_MIN_OVERALL_BALLS = 30, 5, 100
BOWL_MIN_PRESSURE_BALLS, BOWL_MIN_PRESSURE_MATCHES, BOWL_MIN_OVERALL_BALLS = 24, 5, 60
PRIOR_BALLS = 30

# IPL 2027 retained players across ALL franchises, based on the official IPL
# retention announcement dated 18 Nov 2026. The mapping file supplies the
# 2026 IPL player IDs, allowing the filter to be ID-based rather than name-only.
RETAINED_NAMES = {
    # CSK
    "Akeal Hosein", "Anshul Kamboj", "Ayush Mhatre", "Dewald Brevis",
    "Gurjapneet Singh", "Jack Edwards", "Jamie Overton", "Kartik Sharma",
    "Liam Livingstone", "Mukesh Choudhary", "Noor Ahmad", "Rahul Chahar",
    "Ramakrishna Ghosh", "Ruturaj Gaikwad", "Sanju Samson", "Sarfaraz Khan",
    "Shivam Dube", "Shreyas Gopal", "Urvil Patel",
    # DC
    "Ashutosh Sharma", "Auqib Nabi", "Axar Patel", "David Miller",
    "KL Rahul", "Kyle Jamieson", "Lungi Ngidi", "Madhav Tiwari",
    "Mukesh Kumar", "Prithvi Shaw", "Rishabh Pant", "Sameer Rizvi",
    "Tripurana Vijay", "Tristan Stubbs", "Vipraj Nigam", "Zeeshan Ansari",
    # GT
    "Anuj Rawat", "Ashok Sharma", "Glenn Phillips", "Gurnoor Brar",
    "Harshal Patel", "Jason Holder", "Jayant Yadav", "Jos Buttler",
    "Kagiso Rabada", "Kumar Kushagra", "Manav Suthar", "Mohammed Siraj",
    "Nishant Kumar Sindhu", "Prasidh Krishna", "Rahul Tewatia", "Rashid Khan",
    "Sai Kishore", "Sai Sudharsan", "Shubman Gill", "Tom Banton", "Washington Sundar",
    # KKR
    "Angkrish Raghuvanshi", "Anukul Roy", "Cameron Green", "Finn Allen",
    "Harshit Rana", "Kartik Tyagi", "Manish Pandey", "Prashant Solanki",
    "Ramandeep Singh", "Rinku Singh", "Rovman Powell", "Sarthak Ranjan",
    "Saurabh Dubey", "Sunil Narine", "Tim Seifert", "Vaibhav Arora",
    "Varun Chakravarthy",
    # LSG
    "Abdul Samad", "Aiden Markram", "Akash Singh", "Akshat Raghuwanshi",
    "Arjun Tendulkar", "Arshin Kulkarni", "Ayush Badoni", "Digvesh Rathi",
    "Josh Inglis", "Kuldeep Yadav", "Mayank Yadav", "Mitchell Marsh",
    "Mohsin Khan", "Mukul Choudhary", "Nicholas Pooran", "Prince Yadav", "Sam Curran", "Shahbaz Ahmed",
    # MI
    "Arshdeep Singh", "Ashwani Kumar", "Corbin Bosch", "Jasprit Bumrah",
    "Mitchell Santner", "Naman Dhir", "Quinton de Kock", "Raghu Sharma",
    "Raj Bawa", "Rohit Sharma", "Ryan Rickelton", "Shardul Thakur",
    "Sherfane Rutherford", "Suryakumar Yadav", "Tilak Varma", "Will Jacks",
    # PBKS
    "Azmatullah Omarzai", "Ben Dwarshuis", "Cooper Connolly", "Hardik Pandya",
    "Harnoor Pannu", "Harpreet Brar", "Marco Jansen", "Marcus Stoinis",
    "Musheer Khan", "Prabhsimran Singh", "Priyansh Arya", "Shashank Singh",
    "Shreyas Iyer", "Suryansh Shedge", "Vishnu Vinod", "Vyshak Vijaykumar",
    "Xavier Bartlett", "Yash Thakur", "Yuzvendra Chahal",
    # RR
    "Riyan Parag", "Yashasvi Jaiswal", "Vaibhav Sooryavanshi", "Ravindra Jadeja",
    "Dhruv Jurel", "Jofra Archer", "Aman Rao Perala", "Brijesh Sharma",
    "Nandre Burger", "Dasun Shanaka", "Donovan Ferreira", "Lhuan-dre Pretorius",
    "Ravi Bishnoi", "Sandeep Sharma", "Shubham Dubey", "Sushant Mishra",
    "Yash Raj Punja", "Mohammed Shami", "Manimaran Siddharth",
    # RCB
    "Abhinandan Singh", "Bhuvneshwar Kumar", "Devdutt Padikkal", "Jacob Duffy",
    "Jitesh Sharma", "Jordan Cox", "Josh Hazlewood", "Kanishk Chouhan",
    "Krunal Pandya", "Nuwan Thushara", "Phil Salt", "Rajat Patidar",
    "Rasikh Salam", "Romario Shepherd", "Suyash Sharma", "Tim David",
    "Venkatesh Iyer", "Vicky Ostwal", "Virat Kohli", "Yash Dayal",
    # SRH
    "Abhishek Sharma", "Aniket Verma", "Brydon Carse", "Eshan Malinga",
    "Harsh Dubey", "Heinrich Klaasen", "Ishan Kishan", "Khaleel Ahmed",
    "Krains Fuletra", "Nathan Ellis", "Nitish Kumar Reddy", "Pat Cummins",
    "Praful Hinge", "Sakib Hussain", "Salil Arora", "Shivang Kumar",
    "Smaran Ravichandran", "T. Natarajan"
}

# Known retired / opted-out players are not auction targets even if absent from
# the retained list. Faf was also explicitly excluded by the user.
INELIGIBLE_NAMES = {"Kieron Pollard", "Faf du Plessis", "Faf", "Mahendra Singh Dhoni",
                    "MS Dhoni", "Travis Head", "Jacob Bethell"}

mapping = load_one("competition_team_player_mapping")
current_roster = mapping[mapping["comp_id"] == 3298].copy()
# Name aliases between the player mapping and the official retention notice.
retained_aliases = {"KL Rahul": "Lokesh Rahul", "Tilak Varma": "Namboori Tilak Varma",
                    "Lhuan-dre Pretorius": "Lhuan Pretorius", "Vyshak Vijaykumar": "Vyshak Kumar",
                    "Auqib Nabi": "Auqib Dar", "Yash Raj Punja": "Yash Punja",
                    "T. Natarajan": "T Natarajan", "Pat Cummins": "Patrick Cummins",
                    "Nishant Kumar Sindhu": "Nishant Sindhu", "Gurnoor Brar": "Gurnoor Singh Brar",
                    "Rasikh Salam": "Rasikh Dar", "Raj Bawa": "Raj Angad Bawa",
                    "Nitish Reddy": "Nitish Kumar Reddy", "Arshad Khan": "Mohd. Arshad Khan",
                    "William Jacks": "Will Jacks", "Ravichandran Smaran": "Smaran Ravichandran"}
retained_lookup = set(RETAINED_NAMES)
retained_lookup.update(retained_aliases.keys())
retained_lookup.update(retained_aliases.values())
retained_ids = set(current_roster.loc[current_roster.player_name.isin(retained_lookup), "player_id"])
# Candidate pool is restricted to players in the 2026 IPL squad mapping who
# were not retained for 2027. Registration is not yet verified.
released_candidate_ids = set(current_roster.loc[~current_roster.player_id.isin(retained_ids), "player_id"])
ineligible_ids = set(current_roster.loc[current_roster.player_name.isin(INELIGIBLE_NAMES), "player_id"])
released_candidate_ids -= ineligible_ids

meta = matches.merge(
    competitions[["comp_id", "comp_name", "match_format", "season"]],
    on="comp_id", how="left", validate="many_to_one"
)
d = ball.merge(
    meta[["match_id", "comp_id", "comp_name", "season", "revised_target"]],
    on="match_id", how="left", validate="many_to_one"
)
d = d[d["comp_id"].isin(IPL_IDS) & d["innings"].isin([1, 2])].copy()

numeric = ["runs", "extras1_runs", "extras_2_runs", "is_wicket",
           "is_four", "is_six", "overid", "innings_ball_num"]
for c in numeric:
    d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

d["total_runs"] = d["runs"] + d["extras1_runs"] + d["extras_2_runs"]
d["legal"] = ~d["extras_type"].isin(["wide", "Overstep"])
d["legal"] = d["legal"].astype(bool)
d = d.sort_values(["match_id", "innings", "innings_ball_num"]).copy()
keys = ["match_id", "innings"]
d["score_before"] = d.groupby(keys)["total_runs"].cumsum() - d["total_runs"]
d["wickets_before"] = d.groupby(keys)["is_wicket"].cumsum() - d["is_wicket"]
d["legal_before"] = d.groupby(keys)["legal"].transform(
    lambda s: s.astype(int).cumsum().shift(1, fill_value=0)
)
inn = d.groupby(keys).agg(innings_runs=("total_runs", "sum"),
                          legal_balls=("legal", "sum")).reset_index()
first = inn[inn["innings"] == 1][["match_id", "innings_runs"]].rename(
    columns={"innings_runs": "first_innings_runs"})
d = d.merge(first, on="match_id", how="left")
# Standard 20-over regulation limit. Revised targets are honoured where supplied.
d["target"] = d["revised_target"].where(
    d["revised_target"].notna(), d["first_innings_runs"] + 1
)
d["balls_remaining"] = (120 - d["legal_before"]).clip(lower=1)
d["rrr"] = ((d["target"] - d["score_before"]).clip(lower=0) /
           d["balls_remaining"]) * 6

def pressure_flag(frame, scenario):
    if scenario == "permissive":
        chase, late, late_wkts, first_wkts = 9.5, 7.5, 2, (2, 3, 4)
    elif scenario == "base":
        chase, late, late_wkts, first_wkts = 10.0, 8.0, 3, (3, 4, 5)
    elif scenario == "strict":
        chase, late, late_wkts, first_wkts = 11.0, 9.0, 4, (4, 5, 6)
    else:
        raise ValueError(scenario)
    over, inn_no, wkts, rrr = frame.overid, frame.innings, frame.wickets_before, frame.rrr
    chasing = (inn_no == 2) & ((rrr >= chase) |
               (over.between(16, 20) & (rrr >= late) & (wkts >= late_wkts)))
    batting_first = (inn_no == 1) & (
        (over.between(1, 10) & (wkts >= first_wkts[0])) |
        (over.between(11, 15) & (wkts >= first_wkts[1])) |
        (over.between(16, 20) & (wkts >= first_wkts[2]))
    )
    return chasing | batting_first

for sc in ["permissive", "base", "strict"]:
    d[f"pressure_{sc}"] = pressure_flag(d, sc)

# The source contains 'other' extras; exclude those rows from player-ball denominators.
d["batter_ball"] = (~d["extras_type"].isin(["wide", "other", "Penalty"])).astype(int)
d["boundary"] = ((d["is_four"] == 1) | (d["is_six"] == 1)).astype(int)
d["bowler_runs"] = d["total_runs"] - np.where(
    d["extras_type"].isin(["byes", "legbyes", "Penalty"]),
    d["extras1_runs"] + d["extras_2_runs"], 0
)
credited = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}
d["bowler_wicket"] = (
    (d["is_wicket"] == 1) & d["dismissal_type"].isin(credited)
).astype(int)
d["dot"] = (d["total_runs"] == 0).astype(int)

def conf(balls, matches):
    return np.sqrt((1 - np.exp(-balls / 60)) * (1 - np.exp(-matches / 10)))

def pct(s, high=True):
    return s.rank(pct=True, ascending=not high, method="average") * 100

def batter_table(sc):
    p = d[d[f"pressure_{sc}"] & d["legal"]].copy()
    o = d[d["legal"]].copy()
    pg = p.groupby(["striker_id", "striker_name"]).agg(
        pressure_legal_balls=("legal", "sum"), pressure_matches=("match_id", "nunique"),
        pressure_runs=("runs", "sum"), pressure_balls=("batter_ball", "sum"),
        pressure_dismissals=("is_wicket", "sum"), pressure_boundaries=("boundary", "sum"))
    og = o.groupby(["striker_id", "striker_name"]).agg(
        overall_legal_balls=("legal", "sum"), overall_runs=("runs", "sum"),
        overall_balls=("batter_ball", "sum"))
    s = pg.join(og, how="inner").reset_index()
    s = s[(s.pressure_legal_balls >= BAT_MIN_PRESSURE_BALLS) &
          (s.pressure_matches >= BAT_MIN_PRESSURE_MATCHES) &
          (s.overall_legal_balls >= BAT_MIN_OVERALL_BALLS)].copy()
    s["pressure_sr"] = 100*s.pressure_runs/s.pressure_balls.replace(0, np.nan)
    s["pressure_average"] = s.pressure_runs/s.pressure_dismissals.replace(0, np.nan)
    s["overall_sr"] = 100*s.overall_runs/s.overall_balls.replace(0, np.nan)
    s["shrunk_sr"] = 100*(s.pressure_runs + PRIOR_BALLS*s.overall_sr/100)/(s.pressure_balls+PRIOR_BALLS)
    s["sr_uplift"] = s.shrunk_sr-s.overall_sr
    s["boundary_rate"] = 100*s.pressure_boundaries/s.pressure_balls.replace(0,np.nan)
    s["confidence"] = conf(s.pressure_legal_balls, s.pressure_matches)
    s["score"] = (.35*pct(s.pressure_sr)+.20*pct(s.pressure_average)+
                  .20*pct(s.sr_uplift)+.10*pct(s.boundary_rate)+.15*pct(s.confidence))
    return s

def bowler_table(sc):
    p = d[d[f"pressure_{sc}"] & d["legal"] & (d["extras_type"] != "other")].copy()
    o = d[d["legal"] & (d["extras_type"] != "other")].copy()
    pg = p.groupby(["bowler_id", "bowler_name"]).agg(
        pressure_legal_balls=("legal", "sum"), pressure_matches=("match_id", "nunique"),
        pressure_runs=("bowler_runs", "sum"), pressure_wickets=("bowler_wicket", "sum"),
        pressure_dots=("dot", "sum"))
    og = o.groupby(["bowler_id", "bowler_name"]).agg(
        overall_legal_balls=("legal", "sum"), overall_runs=("bowler_runs", "sum"))
    s = pg.join(og, how="inner").reset_index()
    s = s[(s.pressure_legal_balls >= BOWL_MIN_PRESSURE_BALLS) &
          (s.pressure_matches >= BOWL_MIN_PRESSURE_MATCHES) &
          (s.overall_legal_balls >= BOWL_MIN_OVERALL_BALLS)].copy()
    s["pressure_economy"] = 6*s.pressure_runs/s.pressure_legal_balls
    s["pressure_wickets_per_over"] = 6*s.pressure_wickets/s.pressure_legal_balls
    s["pressure_dot_rate"] = 100*s.pressure_dots/s.pressure_legal_balls
    s["overall_economy"] = 6*s.overall_runs/s.overall_legal_balls
    s["economy_improvement"] = s.overall_economy-s.pressure_economy
    s["confidence"] = conf(s.pressure_legal_balls, s.pressure_matches)
    s["score"] = (.30*pct(s.pressure_economy,False)+.25*pct(s.pressure_wickets_per_over)+
                  .15*pct(s.pressure_dot_rate)+.15*pct(s.economy_improvement)+.15*pct(s.confidence))
    return s

def consensus(fn, id_col, name_col):
    tables = {}
    for sc in ["permissive", "base", "strict"]:
        t = fn(sc).sort_values("score", ascending=False).reset_index(drop=True)
        t[f"{sc}_rank"] = np.arange(1, len(t)+1)
        tables[sc] = t
    common = set(tables["permissive"][id_col])
    common &= set(tables["base"][id_col])
    common &= set(tables["strict"][id_col])
    c = None
    for sc,t in tables.items():
        part = t[t[id_col].isin(common)][[id_col,name_col,f"{sc}_rank"]]
        c = part if c is None else c.merge(part[[id_col,f"{sc}_rank"]], on=id_col)
    if c is None or c.empty:
        return pd.DataFrame(), tables
    ranks = ["permissive_rank","base_rank","strict_rank"]
    c["mean_rank"] = c[ranks].mean(axis=1)
    c["rank_dispersion"] = c[ranks].std(axis=1)
    c = c.sort_values(["mean_rank","rank_dispersion"]).reset_index(drop=True)
    c["consensus_rank"] = np.arange(1,len(c)+1)
    return c, tables

bat_c, bat_s = consensus(batter_table, "striker_id", "striker_name")
bowl_c, bowl_s = consensus(bowler_table, "bowler_id", "bowler_name")

def final_table(cons, tables, id_col, name_col, role):
    if cons.empty: return pd.DataFrame()
    base = tables["base"]
    out = cons.merge(base, on=[id_col,name_col], how="left", suffixes=("","_base"))
    out = out[out[id_col].isin(released_candidate_ids)].copy()
    out["availability_status"] = "Not retained for IPL 2027; auction registration unconfirmed"
    out = out.sort_values(["mean_rank","rank_dispersion"]).head(10)
    out.insert(0,"scouting_rank",range(1,len(out)+1))
    return out

bat_final = final_table(bat_c,bat_s,"striker_id","striker_name","batter")
bowl_final = final_table(bowl_c,bowl_s,"bowler_id","bowler_name","bowler")

bat_cols = ["scouting_rank","striker_name","availability_status","pressure_legal_balls","pressure_matches",
            "pressure_runs","pressure_sr","pressure_average","overall_sr","sr_uplift",
            "boundary_rate","score","mean_rank","rank_dispersion"]
bowl_cols = ["scouting_rank","bowler_name","availability_status","pressure_legal_balls","pressure_matches",
             "pressure_wickets","pressure_economy","pressure_wickets_per_over",
             "pressure_dot_rate","overall_economy","economy_improvement",
             "score","mean_rank","rank_dispersion"]

bat_final[bat_cols].round(2).to_csv(ROOT/"RR_2027_Batting_Scouting_Rankings.csv",index=False)
bowl_final[bowl_cols].round(2).to_csv(ROOT/"RR_2027_Bowling_Scouting_Rankings.csv",index=False)
with pd.ExcelWriter(ROOT/"RR_2027_Scouting_Rankings.xlsx") as x:
    bat_final[bat_cols].round(2).to_excel(x,sheet_name="Batting",index=False)
    bowl_final[bowl_cols].round(2).to_excel(x,sheet_name="Bowling",index=False)

print("IPL deliveries analysed:",len(d))
print("IPL matches analysed:",d.match_id.nunique())
print("Batting shortlist: 2027 non-retained candidates; registration unconfirmed")
print(bat_final[bat_cols].head(10).round(2).to_string(index=False))
print("\\nBowling shortlist: non-retained scouting pool; auction registration not verified")
print(bowl_final[bowl_cols].head(10).round(2).to_string(index=False))
print("\\nExports: RR_2027_Batting_Scouting_Rankings.csv, RR_2027_Bowling_Scouting_Rankings.csv, RR_2027_Scouting_Rankings.xlsx")
