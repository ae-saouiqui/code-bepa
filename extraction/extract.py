import riegeli
import glob
import contest_problem_pb2
import io
import json
import sys
from pathlib import Path


def get_records(input_pattern:str):
    filenames = glob.glob(input_pattern)
    if not filenames :
        raise FileNotFoundError(f"No file found  with this pattern {input_pattern}")
    
    for filename in filenames :
        reader = riegeli.RecordReader(io.FileIO(filename,mode='rb'),)
        for record in reader.read_messages(contest_problem_pb2.ContestProblem):
            yield record

def get_problem_id(record):
    source = contest_problem_pb2.ContestProblem.Source.Name(
        record.source
    )

    identity = f"{source}_{record.name}"
    return identity.lower()

def extract_fields(input_pattern:str):
    for record in get_records(input_pattern):
        problem_id = get_problem_id(record)
        for index,solution in enumerate(record.solutions):
            print("Record : ",index)
            problem = {
                "problem_id": problem_id,
                "solution_id" : f"{problem_id}_{index}",
                "name" : record.name,
                "description" :record.description,
                "difficulty" : contest_problem_pb2.ContestProblem.Difficulty.Name(record.difficulty),
                "solution" : solution.solution,
                "language" : contest_problem_pb2.ContestProblem.Solution.Language.Name(solution.language)
                }

            yield problem


def load_and_extract(input_pattern:str,output_file:Path):
    print("Start Extraction")
    with output_file.open("w",encoding="utf-8") as file:
        for record in extract_fields(input_pattern) :
            json.dump(record,file,ensure_ascii=False)
            file.write("\n")

    print("Fields extracted succefully")


def main():

    if len(sys.argv)!=3 : 
        print(
            """Usage extract <input_pattern> <output file>
                  
                  <input_pattern> : file or a regex pattern .
                  <output_file> : A jsonl file
            """,
            file=sys.stderr
        )
        sys.exit(1)

    input_pattern = sys.argv[1]
    output_file = Path(sys.argv[2])
    # checking if the file a json extension 
    if output_file.suffix != ".jsonl":
        print("<output_file> must be of type jsonl",file=sys.stderr)
        sys.exit(1)
    try :
        load_and_extract(input_pattern,output_file)
    except Exception : 
        if output_file.exists():
            output_file.unlink()
        raise 


if __name__ == "__main__" : 
    main()
