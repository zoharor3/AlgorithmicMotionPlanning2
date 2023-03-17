import argparse
from MapEnvironment import MapEnvironment
from TaskInspectionPlanner import TaskInspectionPlanner
from utils import write_plan_stats

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='script for testing planners')
    parser.add_argument('-map', '--map', type=str, default='map_plan_p1.json', help='Json file name containing all map information')
    parser.add_argument('-coverage', '--coverage', type=float, default=0.5, help='percentage of points to inspect (inspection planning)')
    args = parser.parse_args()

    # prepare the map
    planning_env = MapEnvironment(json_file=args.map)

    # setup and execute the planner
    planner = TaskInspectionPlanner(planning_env=planning_env, coverage=args.coverage)
    plan, plan_timestamps, path_coverage, path_cost, computation_time = planner.plan()

    # visualize the final path
    planner.planning_env.visualize_plan(plan=plan, plan_timestamps=plan_timestamps)

    # print 
    write_plan_stats(path_coverage=path_coverage, path_cost=path_cost, computation_time=computation_time)
