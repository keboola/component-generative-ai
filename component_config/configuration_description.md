### Parameters:

#### AI Service Provider: OpenAI

- **API Key (`#api_token`):** Obtain your API key from the [OpenAI platform settings](https://platform.openai.com/account/api-keys).

#### AI Service Provider: Azure OpenAI

- **API Key (`#api_token`)**
- **API Base (`api_base`)**
- **Deployment ID (`deployment_id`)**
- **API Version (`api_version`):** API version used to call the Completions endpoint. Check the list of supported API versions in the [Microsoft Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference?WT.mc_id=AZ-MVP-5004796#completions).

For information on retrieving your API key, API Base, and Deployment ID, refer to the [Microsoft Azure documentation](https://learn.microsoft.com/cs-cz/azure/ai-services/openai/quickstart?tabs=command-line&pivots=programming-language-python#retrieve-key-and-endpoint).

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