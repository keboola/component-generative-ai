"""
Template Component main class.

"""
import csv
import dataclasses
import logging
import time
from typing import List
import json
import os

import pystache as pystache
from keboola.component.base import ComponentBase, sync_action
from keboola.component.sync_actions import ValidationResult, MessageType
from keboola.component.dao import TableDefinition
from keboola.component.exceptions import UserException
from keboola.utils.helpers import comma_separated_values_to_list

from configuration import Configuration
from client.AIClient import OpenAIClient, AIClientException

# configuration variables
RESULT_COLUMN_NAME = 'result_value'
KEY_API_TOKEN = '#api_token'
KEY_SLEEP = 'sleep'
KEY_PROMPT = 'prompt'
KEY_DESTINATION = 'destination'
KEY_STORE_RESULTS_ON_FAILURE = 'store_results_on_failure'

# list of mandatory parameters => if some is missing,
# component will fail with readable message on initialization.
REQUIRED_PARAMETERS = [KEY_API_TOKEN, KEY_PROMPT, KEY_DESTINATION]
REQUIRED_IMAGE_PARS = []


class Component(ComponentBase):
    """
        Extends base class for general Python components. Initializes the CommonInterface
        and performs configuration validation.

        For easier debugging the data folder is picked up by default from `../data` path,
        relative to working directory.

        If `debug` parameter is present in the `config.json`, the default logger is set to verbose DEBUG mode.
    """

    def __init__(self):
        super().__init__()
        self.api_key = None
        self.model_options = None
        self.inference_function = None
        self.store_results_on_failure = None
        self.input_keys = None
        self.sleep_time = None
        self.queue_v2 = None
        self.model = None
        self._configuration = None
        self.failed_requests = 0

    def run(self):
        """
        Main execution code
        """
        self.init_configuration()

        client = self.get_client()

        input_table, out_table = self.prepare_tables()

        logging.info('Querying OpenAI.')
        with open(input_table.full_path, 'r') as input_file:
            reader = csv.DictReader(input_file)
            with open(out_table.full_path, 'w+') as out_file:
                writer = csv.DictWriter(out_file, fieldnames=out_table.primary_key + [RESULT_COLUMN_NAME])
                writer.writeheader()
                for row in reader:
                    prompt = self._build_prompt(self.input_keys, row)
                    result = client.infer(prompt, **self.model_options)

                    if result:
                        writer.writerow(self._build_output_row(out_table.primary_key, row, result))
                    time.sleep(self.sleep_time)

        self.write_manifest(out_table)

        if self.failed_requests > 0:
            if self.store_results_on_failure:
                if self.queue_v2:
                    self.add_flag_to_manifest()
                raise UserException(f"Component has failed to process {self.failed_requests} records.")
        else:
            logging.info("All rows processed.")

    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)

        self.sleep_time = self._configuration.sleep
        self.input_keys = self._get_input_keys(self._configuration.prompt)

        self.queue_v2 = False
        self.store_results_on_failure = self._configuration.destination.store_results_on_failure
        if self.store_results_on_failure:
            self.queue_v2 = 'queuev2' in os.environ.get('KBC_PROJECT_FEATURE_GATES', '')
            if self.queue_v2:
                logging.info("Component will try to save results even if some queries fail.")
            else:
                logging.warning("Running on old queue, results cannot be stored on failure.")

        self.api_key = self._configuration.pswd_api_token

        self.model = self._configuration.custom_model or self._configuration.predefined_model
        logging.info(f"The component is using the model : {self.model}")

        self.model_options = dataclasses.asdict(self._configuration.additional_options)

    def get_client(self):
        return OpenAIClient(self.api_key, self.model)

    def prepare_tables(self):
        input_table = self._get_input_table()
        if missing_keys := [key for key in self.input_keys if key not in input_table.columns]:
            raise UserException(f'The columns "{missing_keys}" need to be present in the input data!')

        out_table = self._build_out_table()

        if missing_keys := [t for t in out_table.primary_key if t not in input_table.columns]:
            raise UserException(f'Some specified primary keys are not in the input table: {missing_keys}')

        return input_table, out_table

    @staticmethod
    def _get_sleep(params):
        sleep_time = float(params.get(KEY_SLEEP, 0))
        logging.info(f"Sleep time set to {sleep_time} seconds." if sleep_time else "Sleep time not set.")
        return sleep_time

    @staticmethod
    def _build_output_row(primary_key: List[str], input_row: dict, result: str):
        row = dict()
        for k in primary_key:
            row[k] = input_row[k]
        row['result_value'] = result.strip()
        return row

    def _build_out_table(self) -> TableDefinition:
        destination_config = self.configuration.parameters['destination']
        out_table_name = f"{destination_config['output_table_name']}.csv"

        # backward compatibility with the old string element
        if destination_config.get('primary_keys', None):
            primary_key = comma_separated_values_to_list(destination_config.get('primary_keys', None))
        else:
            primary_key = destination_config.get('primary_keys_array', [])

        incremental_load = destination_config.get('incremental_load', False)
        return self.create_out_table_definition(out_table_name, primary_key=primary_key,
                                                incremental=incremental_load)

    @staticmethod
    def _get_input_keys(prompt: str):
        template = pystache.parse(prompt, delimiters=('[[', ']]'))
        keys = [token.key for token in template._parse_tree if hasattr(token, "key")]  # noqa
        if len(keys) < 1:
            raise UserException('You must provide at least one input placeholder. 0 were found.')
        return keys

    def _build_prompt(self, input_keys: List[str], row: dict):
        prompt = self.configuration.parameters['prompt']
        for input_key in input_keys:
            prompt = prompt.replace('[[' + input_key + ']]', row[input_key])
        return prompt

    def _get_input_table(self) -> TableDefinition:
        if not self.get_input_tables_definitions():
            raise UserException("No input table specified. Please provide one input table in the input mapping!")
        return self.get_input_tables_definitions()[0]

    def add_flag_to_manifest(self):
        manifests = [x for x in os.listdir(self.tables_out_path) if x.endswith('.manifest')]
        if manifests:
            for filename in manifests:
                path = os.path.join(self.tables_out_path, filename)

                with open(path, 'r') as f:
                    data = json.load(f)
                    data['write_always'] = True

                with open(path, 'w') as f:
                    json.dump(data, f)

    @sync_action('list_pkeys')
    def list_table_columns(self):
        """
        Sync action to fill the UI element of primary keys.

        Returns:

        """
        input_table = self._get_input_table()
        return [{"value": c, "label": c} for c in input_table.columns]

    @sync_action('testQuery')
    def test_query(self) -> ValidationResult:
        self.init_configuration()

        client = self.get_client()

        input_table, out_table = self.prepare_tables()

        results = []
        with open(input_table.full_path, 'r') as input_file:
            reader = csv.DictReader(input_file)

            row_count = 0
            for row in reader:
                prompt = self._build_prompt(self.input_keys, row)
                result = client.infer(prompt, **self.model_options)

                if result:
                    results.append(self._build_output_row(out_table.primary_key, row, result))
                time.sleep(self.sleep_time)

                row_count += 1
                if row_count == 3:
                    break  # Stop processing after three rows

        markdown = self.create_markdown_table(results)
        return ValidationResult(markdown, MessageType.SUCCESS)

    @staticmethod
    def create_markdown_table(data):
        if not data:
            return ""
        headers = list(data[0].keys())
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in data:
            row_values = [str(row[header]) for header in headers]
            table += "| " + " | ".join(row_values) + " |\n"
        return table


"""
        Main entrypoint
"""
if __name__ == "__main__":
    try:
        comp = Component()
        # this triggers the run method by default and is controlled by the configuration.action parameter
        comp.execute_action()
    except UserException as exc:
        logging.exception(exc)
        exit(1)
    except Exception as exc:
        logging.exception(exc)
        exit(2)
