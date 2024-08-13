# Generative AI

This component enables you to query OpenAI, Azure OpenAI, and Llama 2 (coming soon) with data provided from your KBC project.

- [TOC]

---

## Configuration

### Parameters:

#### AI Service Provider: OpenAI

- **API Key (`#api_token`):** Obtain your API key from the [OpenAI platform settings](https://platform.openai.com/account/api-keys).

#### AI Service Provider: Azure OpenAI

- **API Key (`#api_token`)**
- **API Base (`api_base`)**
- **Deployment ID (`deployment_id`)**
- **API Version (`api_version`):** API version used to call the Completions endpoint. Check the list of supported API versions in the [Microsoft Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference?WT.mc_id=AZ-MVP-5004796#completions).

For information on retrieving your API key, API Base, and Deployment ID, refer to the [Microsoft Azure documentation](https://learn.microsoft.com/cs-cz/azure/ai-services/openai/quickstart?tabs=command-line&pivots=programming-language-python#retrieve-key-and-endpoint).

#### AI Service Provider: Google

- Only works on US AWS stack – connection.keboola.com
- Default API Key for this service is provided by Keboola. 

#### AI Service Provider: Hugging Face

- API key is Provided by Keboola if one of the predefined models is used.
- If you want to use a custom model, you need to provide your own API key.

- **API Key (`#api_token`)**
- **API Key (`endpoint_url`)** - You can use any endpoint url from your <a href="https://ui.endpoints.huggingface.co/">HuggingFace web console.</a> or a serverless deployment.

### Other options:

- **Model (`prompt`)** You can use the sync action to load available models for your account.
- **Prompt (`prompt`):** The prompt and data input pattern. Use the placeholder [[INPUT_COLUMN]] to refer to the input column. The input table must contain the referenced column.
- **Incremental Load (`incremental load`):** If enabled, the table will update rather than be overwritten. Tables with primary keys will update rows, whereas those without a primary key will append rows.
- **Output Table Name (`output_table_name`)**
- **Primary Keys Array (`primary_keys_array`):** You can input multiple columns separated by commas, e.g., id, other_id. Selecting incremental loads allows for table updates if a primary key is set. A primary key can have multiple columns, and the primary key of an existing table is immutable.
- **Predefined Model (`predefined_model`):** The model that will generate the completion. [Learn more](https://beta.openai.com/docs/models).

**Additional Options:**

- **Max Tokens (`max_tokens`):** Maximum number of tokens for the completion. The token count of your prompt plus `max_tokens` should not exceed the model's context length. Most models support up to 2048 tokens, with the newest ones supporting 4096.
- **Temperature (`temperature`):** Sampling temperature between [0-1]. Higher values result in riskier outputs. Use 0.9 for creativity, and 0 for well-defined answers.
- **Top P (`top_p`):** Nucleus sampling, where only tokens with top_p probability mass are considered. For instance, 0.1 means only the top 10% probability mass tokens are evaluated.
- **Frequency Penalty (`frequency_penalty`):** A number between -2.0 and 2.0. Positive values penalize frequently occurring tokens in the current text, reducing repetition.
- **Presence Penalty (`presence_penalty`):** A number between -2.0 and 2.0. Positive values penalize tokens already present in the text, encouraging diverse topics.
- **Request Timeout (`request_timeout`):** Seconds to wait for API to respond. This is a workaround for OpenAI API not responding sometimes.

---

### Component Configuration Example

**Generic configuration**

```json
{
  "parameters": {
    "#api_token": "secret_api_token",
    "sleep": 5
  }
}
```

**Row configuration**

```json
{
  "parameters": {
    "prompt": "Extract keywords from this text:\n\n\"\"\"\n[[INPUT]]\n\"\"\"",
    "model_type": "predefined",
    "destination": {
      "incremental_load": true,
      "output_table_name": "keywords_test",
      "primary_keys_array": [
        "message_id"
      ]
    },
    "predefined_model": "text-davinci-002",
    "additional_options": {
      "top_p": 1,
      "max_tokens": 100,
      "temperature": 0.6,
      "presence_penalty": 0,
      "frequency_penalty": 0
    }
  }
}
```


# Development

If required, change local data folder (the `CUSTOM_FOLDER` placeholder) path to your custom path in
the `docker-compose.yml` file:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    volumes:
      - ./:/code
      - ./CUSTOM_FOLDER:/data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone this repository, init the workspace and run the component with following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git clone git@bitbucket.org:kds_consulting_team/kds_consulting_team/kds-team.generative-ai.git kds-team.generative-ai
cd kds-team.app-open-ai
docker-compose build
docker-compose run --rm dev
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the test suite and lint check using this command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
docker-compose run --rm test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integration
===========

For information about deployment and integration with KBC, please refer to the
[deployment section of developers documentation](https://developers.keboola.com/extend/component/deployment/)
