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
from io import StringIO
from itertools import islice


import pystache as pystache
import requests.exceptions
from keboola.component.base import ComponentBase, sync_action
from keboola.component.sync_actions import ValidationResult, MessageType
from keboola.component.dao import TableDefinition
from keboola.component.exceptions import UserException
from kbcstorage.tables import Tables
from kbcstorage.client import Client


from configuration import Configuration
from client.openai_client import OpenAIClient, AzureOpenAIClient
from client.base import AIClientException

# configuration variables
RESULT_COLUMN_NAME = 'result_value'
KEY_API_TOKEN = '#api_token'
KEY_SLEEP = 'sleep'
KEY_PROMPT = 'prompt'
KEY_DESTINATION = 'destination'

# list of mandatory parameters => if some is missing,
# component will fail with readable message on initialization.
REQUIRED_PARAMETERS = [KEY_API_TOKEN, KEY_PROMPT, KEY_DESTINATION]
REQUIRED_IMAGE_PARS = []

PREVIEW_LIMIT = 5
PROMPT_TEMPLATES = 'templates/prompts.json'


class Component(ComponentBase):
    """
        Extends base class for general Python components. Initializes the CommonInterface
        and performs configuration validation.

        For easier debugging the data folder is picked up by default from `../data` path,
        relative to working directory.

        If `debug` parameter is present in the `azure_config.json`, the default logger is set to verbose DEBUG mode.
    """

    def __init__(self):
        super().__init__()
        self.service = None
        self.api_key = None
        # For Azure OpenAI
        self.deployment_id = None
        self.api_base = None
        self.api_version = None

        self.max_token_spend = 0
        self.model_options = None
        self.input_keys = None
        self.sleep_time = None
        self.queue_v2 = None
        self.model = None
        self._configuration = None
        self.failed_requests = 0
        self.tokens_used = 0
        self.token_limit_reached = False

    def run(self):
        """
        Main execution code
        """
        self.init_configuration()

        client = self.get_client()
        inference_function = client.get_inference_function(self.model)

        input_table, out_table = self.prepare_tables()

        with open(input_table.full_path, 'r') as input_file:
            reader = csv.DictReader(input_file)
            with open(out_table.full_path, 'w+') as out_file:
                writer = csv.DictWriter(out_file, fieldnames=out_table.columns)
                writer.writeheader()
                for row in reader:
                    prompt = self._build_prompt(self.input_keys, row)

                    try:
                        result, token_usage = client.infer(self.model, inference_function, prompt,
                                                           **self.model_options)
                        self.tokens_used += token_usage
                    except AIClientException as e:
                        raise UserException(f"The component failed to process request. {e}") from e

                    if result:
                        writer.writerow(self._build_output_row(row, result))
                    else:
                        self.failed_requests += 1

                    if self.max_token_spend != 0 and self.tokens_used >= self.max_token_spend:
                        self.token_limit_reached = True
                        logging.error(f"The token spend limit of {self.max_token_spend} has been reached.")
                        break

                    time.sleep(self.sleep_time)

        self.write_manifest(out_table)

        if self.failed_requests > 0:
            if self.queue_v2:
                self.add_flag_to_manifest()
            raise UserException(f"Component has failed to process {self.failed_requests} records.")
        else:
            if self.token_limit_reached:
                logging.error("Component has been stopped after reaching total token spend limit.")
            else:
                logging.info(f"All rows processed, total token usage = {self.tokens_used}")

    def init_configuration(self):
        self.validate_configuration_parameters(Configuration.get_dataclass_required_parameters())
        self._configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)

        self.sleep_time = self._configuration.sleep

        if self._configuration.max_token_spend > 0:
            self.max_token_spend = self._configuration.max_token_spend
            logging.warning(f"Max token spend has been set to {self.max_token_spend}. If the component reaches "
                            f"this limit, it will exit.")

        self.input_keys = self._get_input_keys(self._configuration.prompt_options.prompt)

        self.queue_v2 = 'queuev2' in os.environ.get('KBC_PROJECT_FEATURE_GATES', '')
        if self.queue_v2:
            logging.info("Component will try to save results even if some queries fail.")
        else:
            logging.warning("Running on old queue, results cannot be stored on failure.")

        self.service = self._configuration.authentication.service
        if self._configuration.authentication.service == "azure_openai":
            self.api_base = self._configuration.authentication.api_base
            self.deployment_id = self._configuration.authentication.deployment_id
            self.api_version = self._configuration.authentication.api_version
        self.api_key = self._configuration.authentication.pswd_api_token

        self.model = self._configuration.model
        logging.info(f"The component is using the model: {self.model}")

        self.model_options = dataclasses.asdict(self._configuration.additional_options)

    def get_client(self):
        if self.service == "openai":
            return OpenAIClient(self.api_key)
        elif self.service == "azure_openai":
            return AzureOpenAIClient(self.api_key, self.api_base, self.deployment_id, self.api_version)
        else:
            raise UserException(f"{self.service} service is not implemented yet.")

    def prepare_tables(self):
        input_table = self._get_input_table()
        if missing_keys := [key for key in self.input_keys if key not in input_table.columns]:
            raise UserException(f'The columns "{missing_keys}" need to be present in the input data!')

        out_table = self._build_out_table(input_table)

        if missing_keys := [t for t in out_table.primary_key if t not in input_table.columns]:
            raise UserException(f'Some specified primary keys are not in the input table: {missing_keys}')

        return input_table, out_table

    @staticmethod
    def _get_sleep(params):
        sleep_time = float(params.get(KEY_SLEEP, 0))
        logging.info(f"Sleep time set to {sleep_time} seconds." if sleep_time else "Sleep time not set.")
        return sleep_time

    @staticmethod
    def _build_output_row(input_row: dict, result: str):
        output_row = input_row.copy()
        output_row[RESULT_COLUMN_NAME] = result.strip()
        return output_row

    def _build_out_table(self, input_table: TableDefinition) -> TableDefinition:
        destination_config = self.configuration.parameters['destination']

        if not (out_table_name := destination_config.get("output_table_name")):
            out_table_name = f"{self.environment_variables.config_row_id}.csv"
        else:
            out_table_name = f"{out_table_name}.csv"

        columns = input_table.columns + [RESULT_COLUMN_NAME]
        primary_key = destination_config.get('primary_keys_array', [])

        incremental_load = destination_config.get('incremental_load', False)
        return self.create_out_table_definition(out_table_name, columns=columns, primary_key=primary_key,
                                                incremental=incremental_load)

    @staticmethod
    def _get_input_keys(prompt: str):
        template = pystache.parse(prompt, delimiters=('[[', ']]'))
        keys = [token.key for token in template._parse_tree if hasattr(token, "key")]  # noqa
        if len(keys) < 1:
            raise UserException('You must provide at least one input placeholder. 0 were found.')
        return keys

    def _build_prompt(self, input_keys: List[str], row: dict):
        prompt = self._configuration.prompt_options.prompt
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

    def estimate_token_usage(self, preview_size: int, table_size: int) -> int:
        """Estimates token usage based on number of tokens used during test_prompt."""
        if preview_size > 0:
            return int((self.tokens_used / preview_size) * table_size)
        raise UserException("Cannot process tables with no rows.")

    @staticmethod
    def create_markdown_table(data):
        if not data:
            return ""
        headers = list(data[0].keys())
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in data:
            row_values = [str(row[header]).replace("\n", " ") for header in headers]
            table += "| " + " | ".join(row_values) + " |\n"
        return table

    def _get_table_preview(self, table_id: str, columns: list[str] = None, limit: int = None) -> list[dict]:
        tables = Tables(self._get_kbc_root_url(), self._get_storage_token())
        try:
            preview = tables.preview(table_id, columns=columns)
        except requests.exceptions.HTTPError as e:
            raise UserException(f"Unable to retrieve table preview: {e}")

        data = []
        csv_reader = csv.DictReader(StringIO(preview))

        if limit is None:
            for row in csv_reader:
                data.append(row)
        else:
            for row in islice(csv_reader, limit):
                data.append(row)

        return data

    def _get_table_size(self, table_id: str) -> int:
        """Returns number of rows for specified table_id."""
        tables = Tables(self._get_kbc_root_url(), self._get_storage_token())
        detail = tables.detail(table_id)
        rows_count = int(detail.get("rowsCount"))
        return rows_count

    def _get_kbc_root_url(self) -> str:
        return f'https://{self.environment_variables.stack_id}' if self.environment_variables.stack_id \
            else "https://connection.keboola.com"

    def _get_storage_source(self) -> str:
        storage_config = self.configuration.config_data.get("storage")
        source = storage_config["input"]["tables"][0]["source"]
        return source

    def _get_storage_token(self) -> str:
        return self.configuration.parameters.get('#storage_token') or self.environment_variables.token

    def _set_tokens_limit(self) -> any:
        if self._configuration.max_token_spend > 0:
            self.max_token_spend = self._configuration.max_token_spend
            logging.warning(f"Max token spend has been set to {self.max_token_spend}. If the component reaches "
                            f"this limit, it will exit.")

    def _get_table_columns(self, table_id: str) -> list:
        client = Client(self._get_kbc_root_url(), self._get_storage_token())
        table_detail = client.tables.detail(table_id)
        columns = table_detail.get("columns")
        if not columns:
            raise UserException(f"Cannot fetch list of columns for table {table_id}")
        return columns

    @sync_action('listPkeys')
    def list_table_columns(self):
        """
        Sync action to fill the UI element for primary keys selection.

        Returns:

        """
        self.init_configuration()
        table_id = self._get_storage_source()
        columns = self._get_table_columns(table_id)
        return [{"value": c, "label": c} for c in columns]

    @sync_action('testPrompt')
    def test_prompt(self) -> ValidationResult:
        """
        Uses table preview from sapi to apply prompt for on a few table rows.
        It uses same functions as the run method. The only exception is replacing newlines with spaces to ensure
        proper formatting for ValidationResult.
        """
        self.init_configuration()

        client = self.get_client()
        inference_function = client.get_inference_function(self.model)

        table_id = self._get_storage_source()
        if len(self.input_keys) > 30:
            raise UserException(f"Test prompt is available only for up to 30 placeholders. "
                                f"You have {len(self.input_keys)} placeholders.")

        table_preview = self._get_table_preview(table_id, columns=self.input_keys, limit=PREVIEW_LIMIT)

        preview_size = len(table_preview)
        table_size = self._get_table_size(table_id)

        if missing_keys := [key for key in self.input_keys if key not in table_preview[0]]:
            raise UserException(f'The columns "{missing_keys}" need to be present in the input data!')

        results = []
        for row in table_preview:
            prompt = self._build_prompt(self.input_keys, row)
            result, token_usage = client.infer(self.model, inference_function, prompt, **self.model_options)
            self.tokens_used += token_usage

            if result:
                output_row = self._build_output_row(row, result)
                output = output_row.get(RESULT_COLUMN_NAME, "")
                results.append(output)

        if results:
            estimated_token_usage = self.estimate_token_usage(preview_size, table_size)

            markdown = self.create_markdown_table(results)
            tokens_used_info = f"\nTokens used during test prompting: {self.tokens_used}"
            token_estimation_info = f"\nEstimated token usage for the whole input table: {estimated_token_usage}"
            markdown += tokens_used_info
            markdown += token_estimation_info
            return ValidationResult(markdown, MessageType.SUCCESS)
        else:
            return ValidationResult("Query returned no data.", MessageType.WARNING)

    @sync_action('getPromptTemplate')
    def get_prompt_template(self) -> ValidationResult:
        configuration: Configuration = Configuration.load_from_dict(self.configuration.parameters)
        template = configuration.prompt_templates.prompt_template

        with open('src/templates/prompts.json', 'r') as json_file:
            templates = json.load(json_file)

        prompt = templates.get(template)
        if not prompt:
            raise UserException(f"Prompt template {template} does not exist!")
        return ValidationResult(prompt, MessageType.SUCCESS)

    @sync_action('listModels')
    def list_models(self):
        authentication = self.configuration.parameters.get("authentication")
        self.service = authentication.get("service")
        self.api_key = authentication.get("#api_token")

        if self.service == "azure_openai":
            self.api_base = authentication.get("api_base")
            self.deployment_id = authentication.get("deployment")
            self.api_version = authentication.get("api_version")

        client = self.get_client()
        models = client.list_models()
        return [{"value": m, "label": m} for m in models]


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
