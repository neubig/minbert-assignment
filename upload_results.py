from __future__ import annotations
import argparse
import json
import os

from explainaboard_api_client.model.system import System
from explainaboard_api_client.model.system_create_props import SystemCreateProps
from explainaboard_api_client.model.system_metadata import SystemMetadata
from explainaboard_api_client.model.system_output_props import SystemOutputProps
from explainaboard_client import Config, ExplainaboardClient
from explainaboard_client.tasks import TaskType
from explainaboard_client.utils import generate_dataset_id


def convert_file(orig_file: str, label_mapping: dict[str, str]) -> str:
    """
    Convert the original file to the new format
    """
    new_file = f'{orig_file}.tmp'
    with open(orig_file, "r") as fin, open(new_file, "w") as fout:
        for line in fin:
            label, text = line.strip().split(" ||| ")
            print(label_mapping[label], file=fout)
    return new_file

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="A command-line tool to upload minbert results "
        "to the ExplainaBoard web interface."
    )
    parser.add_argument(
        "--system_name",
        type=str,
        required=True,
        help="The name of the system",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the system output file",
    )
    parser.add_argument(
        "--dataset", type=str, help="A dataset name from DataLab"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        default="test",
        help="The name of the dataset split to process",
    )
    parser.add_argument(
        "--public", action="store_true", help="Make the uploaded system public"
    )
    args = parser.parse_args()

    # Get environmental variables
    for k in ["EB_API_KEY", "EB_EMAIL", "EB_ANDREW_ID"]:
        if k not in os.environ:
            raise ValueError(f"{k} is not set")
    api_key = os.environ["EB_API_KEY"]
    email = os.environ["EB_EMAIL"]
    andrew_id = os.environ["EB_ANDREW_ID"]

    # Process the file appropriately
    if args.dataset == 'sst':
        label_mapping = {
            '0': 'very negative',
            '1': 'negative',
            '2': 'neutral',
            '3': 'positive',
            '4': 'very positive'
        }
    elif args.dataset == 'cfimdb':
        label_mapping = {
            '0': 'negative',
            '1': 'positive',
        }
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    # Preset values
    task = TaskType.text_classification
    metric_names = ['Accuracy']
    source_language = 'eng'
    target_language = 'eng'
    shared_users = ['neubig@gmail.com']
    system_name = f'anlp_{andrew_id}_{args.system_name}'
    online_split = 'validation' if args.split == 'dev' else args.split

    # Convert file
    new_file = convert_file(args.output, label_mapping)

    # Do upload
    system_output = SystemOutputProps(
        data=new_file,
        file_type='text',
    )
    metadata = SystemMetadata(
        task=task,
        is_private=not args.public,
        system_name=system_name,
        metric_names=metric_names,
        source_language=source_language,
        target_language=target_language,
        dataset_split=online_split,
        shared_users=shared_users,
        system_details={},
    )
    metadata.dataset_metadata_id = generate_dataset_id('cmu_anlp', args.dataset)
    create_props = SystemCreateProps(metadata=metadata, system_output=system_output)
    client_config = Config(
        email,
        api_key,
        environment='main',
    )
    client = ExplainaboardClient(client_config)

    result: System = client.systems_post(create_props)
    try:
        sys_id = result.system_id
        client.systems_get_by_id(sys_id)
        print(f"successfully posted system {system_name} with ID {sys_id}\n"
              f"view the result at https://explainaboard.inspiredco.ai/systems")
    except Exception:
        print(f"failed to post system {system_name}")

    # delete new_file
    os.remove(new_file)

if __name__ == "__main__":
    main()
