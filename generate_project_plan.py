import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_task(id, uid, name, duration_days, start_date, outline_level, predecessor_uid=None, percent_complete=0, is_milestone=False):
    task = ET.Element("Task")
    
    ET.SubElement(task, "UID").text = str(uid)
    ET.SubElement(task, "ID").text = str(id)
    ET.SubElement(task, "Name").text = name
    ET.SubElement(task, "Type").text = "0" # Fixed Units
    ET.SubElement(task, "IsNull").text = "0"
    ET.SubElement(task, "CreateDate").text = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    ET.SubElement(task, "WBS").text = str(id)
    ET.SubElement(task, "OutlineNumber").text = str(id)
    ET.SubElement(task, "OutlineLevel").text = str(outline_level)
    ET.SubElement(task, "Priority").text = "500"
    ET.SubElement(task, "Manual").text = "0" # Auto-scheduled
    ET.SubElement(task, "PercentComplete").text = str(percent_complete)
    
    if is_milestone:
         ET.SubElement(task, "Milestone").text = "1"
         duration_days = 0 
    else:
         ET.SubElement(task, "Milestone").text = "0"

    # Dates: Rough calculation
    end_date = start_date + datetime.timedelta(days=int(duration_days * 7 / 5 * 1.4))
    
    ET.SubElement(task, "Start").text = start_date.strftime("%Y-%m-%dT08:00:00")
    ET.SubElement(task, "Finish").text = end_date.strftime("%Y-%m-%dT17:00:00")
    
    # Duration format
    if is_milestone:
        ET.SubElement(task, "Duration").text = "PT0H0M0S"
    else:
        hours = duration_days * 8
        ET.SubElement(task, "Duration").text = f"PT{hours}H0M0S"
        
    ET.SubElement(task, "DurationFormat").text = "7" 
    
    if predecessor_uid:
        link = ET.SubElement(task, "PredecessorLink")
        ET.SubElement(link, "PredecessorUID").text = str(predecessor_uid)
        ET.SubElement(link, "Type").text = "1" # Finish-to-Start
        ET.SubElement(link, "CrossProject").text = "0"
        ET.SubElement(link, "LinkLag").text = "0"
        ET.SubElement(link, "LagFormat").text = "7"

    return task

def generate_xml(filename):
    root = ET.Element("Project")
    root.set("xmlns", "http://schemas.microsoft.com/project")
    
    ET.SubElement(root, "Name").text = "Guardian GPU Project Plan"
    ET.SubElement(root, "Title").text = "Guardian GPU"
    
    calendars = ET.SubElement(root, "Calendars")
    cal = ET.SubElement(calendars, "Calendar")
    ET.SubElement(cal, "UID").text = "1"
    ET.SubElement(cal, "Name").text = "Standard"
    ET.SubElement(cal, "IsBaseCalendar").text = "1"
    ET.SubElement(cal, "BaseCalendarUID").text = "-1"
    
    tasks = ET.SubElement(root, "Tasks")
    
    # Target: Phase 4 ends ~Feb 2 2026.
    # Back-calculated Start Date: Oct 13, 2025.
    project_start = datetime.datetime(2025, 10, 13, 8, 0, 0)
    
    if project_start.weekday() > 0: 
        days_ahead = 7 - project_start.weekday()
        project_start += datetime.timedelta(days=days_ahead)

    phases = [
        {
            "name": "Phase 1: Research & Analysis",
            "duration_weeks": 4,
            "percent_complete": 100,
            "activities": [
                "Conduct Literature Review on GPU Forensics",
                "Analyze Windows Kernel Tracing (ETW) & NVML capabilities",
                "Define Feature Vectors for Anomaly Detection"
            ],
            "milestone": "Research Complete"
        },
        {
            "name": "Phase 2: System Design",
            "duration_weeks": 4,
            "percent_complete": 100,
            "activities": [
                "Design Split-Architecture (C++ Data Plane vs Python Control Plane)",
                "Define Communication Interface (CSV/Pipe)",
                "Develop a model to accurately differentiate between legit and malicious datasets"
            ],
            "milestone": "Architecture Design Approved"
        },
        {
            "name": "Phase 3: Implementation (Data Plane)",
            "duration_weeks": 4,
            "percent_complete": 100,
            "activities": [
                "Implement GuardianMonitor (C++)",
                "Integrate ETW Context Switch consumers",
                "Build UnifiedLogger for telemetry fusion"
            ],
            "milestone": "Data Plane Ready"
        },
        {
            "name": "Phase 4: Implementation (Control Plane)",
            "duration_weeks": 4,
            "percent_complete": 100,
            "activities": [
                "Develop DecisionOrchestrator (Python)",
                "Train IsolationForest model",
                "Implement Heuristic 'Tier 1' checks"
            ],
            "milestone": "Control Plane Ready (Python Brain Finished)"
        },
        {
            "name": "Phase 5: UI & Dashboard Implementation",
            "duration_weeks": 3,
            "percent_complete": 100,
            "activities": [
                "Design Dashboard Mockups & User Flow",
                "Implement Web Interface (React/Dashboard)",
                "Integrate Real-time Alerts & Toast Notifications"
            ],
            "milestone": "UI & Dashboard Complete"
        },
        {
            "name": "Phase 6: Testing & Closure",
            "duration_weeks": 4,
            "percent_complete": 0,
            "activities": [
                "Perform Attack Simulations (e.g., Cryptojacking)",
                "Measure Latency (Target: <100ms)",
                "Write final thesis and documentation"
            ],
            "milestone": "Project Final Release"
        }
    ]
    
    task_id_counter = 1
    uid_counter = 1
    
    last_phase_uid = None
    
    current_phase_start = project_start
    
    for phase in phases:
        phase_duration_days = phase["duration_weeks"] * 5
        percent = phase.get("percent_complete", 0)
        
        # Phase Summary Task
        phase_task = create_task(task_id_counter, uid_counter, phase["name"], phase_duration_days, current_phase_start, 1, last_phase_uid, percent)
        ET.SubElement(phase_task, "Summary").text = "1"
        tasks.append(phase_task)
        
        current_phase_uid = uid_counter
        task_id_counter += 1
        uid_counter += 1
        
        num_activities = len(phase["activities"])
        subtask_duration = phase_duration_days / num_activities if num_activities > 0 else 1
        
        last_subtask_uid = None
        current_subtask_start = current_phase_start
        
        for activity in phase["activities"]:
            subtask = create_task(task_id_counter, uid_counter, activity, subtask_duration, current_subtask_start, 2, last_subtask_uid, percent)
            ET.SubElement(subtask, "Summary").text = "0"
            tasks.append(subtask)
            
            last_subtask_uid = uid_counter
            task_id_counter += 1
            uid_counter += 1
            
            days_to_add = max(1, int(subtask_duration))
            current_subtask_start += datetime.timedelta(days=int(days_to_add * 1.4))

        # Add Phase Milestone
        milestone_link = last_subtask_uid 
        milestone = create_task(task_id_counter, uid_counter, phase["milestone"], 0, current_subtask_start, 2, milestone_link, percent, is_milestone=True)
        ET.SubElement(milestone, "Summary").text = "0"
        tasks.append(milestone)
        task_id_counter += 1
        uid_counter += 1

        last_phase_uid = current_phase_uid
        current_phase_start += datetime.timedelta(weeks=phase["duration_weeks"])

    tree = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(xmlstr)
    
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    generate_xml("guardian_plan.xml")
