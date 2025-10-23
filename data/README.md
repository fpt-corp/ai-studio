# Dataset Format

---

### Supported Dataset Formats

| Trainer                 | Supported Data Format              | Supported File Format |
| ----------------------- | ---------------------------------- | --------------------- |
| **SFT**                 |  **Alpaca**                        | CSV, JSON, JSONLINES, ZIP, PARQUET |
| **SFT**                 |  **ShareGPT**                      | JSON, JSONLINES, ZIP, PARQUET      |
| **SFT**                 |  **ShareGPT_Image**                | ZIP, PARQUET                       |
| **DPO**                 |  **ShareGPT**                      | JSON, JSONLINES, ZIP, PARQUET      |
| **Pre-training**        |  **Corpus**                        | TXT, JSON, JSONLINES, ZIP, PARQUET |

---

### 1. **Alpaca**

**Alpaca** uses a very simple structure to fine-tune the model with Instruction-following format with input, output pairs for **supervised fine-tuning** tasks. The basic structure includes:

* **instruction**: A string containing the specific task or request that the model needs to perform.
* **input**: A string containing the information that the model needs to process in order to carry out the task.
* **output**: A string representing the result the model should return, generated from processing the instruction and input.

**Structure:**

```json
[
   {
      "instruction": "string",
      "input": "string",
      "output": "string"
   }
]
```

**Example:**

```json
[
  {
    "instruction": "Classify the following into animals, plants, and minerals",
    "input": "Oak tree, copper ore, elephant",
    "output": "Oak tree: Plant\nCopper ore: Mineral\nElephant: Animal"
  },
  {
    "instruction": "Find the area of a circle given its radius.",
    "input": "Radius = 4",
    "output": "The area of a circle with a radius of 4 is equal to 12.5664 square units. This is calculated by using the formula A = πr2, where A is the area, π is roughly equal to 3.1416 and r is the radius of the circle."
  }
]
```

**Sample Dataset:**
[Alpaca Samples](https://github.com/fpt-corp/ai-studio-samples/tree/main/sample-datasets/alpaca)

**Supported file formats:** `.csv`, `.json`, `.jsonlines`, `.zip`, `.parquet`

---

### 2. **ShareGPT**

#### 2.1 Trainer = SFT

**ShareGPT** is designed to represent **multi-turn conversations** between a user and an AI assistant. It is commonly used when training or fine-tuning models for dialogue systems or chatbots that need to handle contextual conversation over multiple turns.

Each data sample consists of a conversations array, where each turn in the chat includes:

* **from**: Who is speaking - must be **human** or **gpt**.
* **value**: The actual message text from that speaker.

Each sample should start with **human** message, followed by **gpt** message.

**Structure:**

```json
[
   {
      "conversations": [
         {
            "from": "human",
            "value": "human message"
         },
         {
            "from": "gpt",
            "value": "gpt message"
         }
      ]
   }
]
```

**Example:**

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "Can you tell me what is the best solution to share files between co workers in a local network? There is a ubuntu server available that can be used to host files."
      },
      {
        "from": "gpt",
        "value": "There are several ways to share files between coworkers on a local network, depending on your specific requirements and the infrastructure you have available..."
      },
      {
        "from": "human",
        "value": "Tell me the best way to use SMB via local network using a Ubuntu server and what configuration is needed on the user side if he's using windows."
      },
      {
        "from": "gpt",
        "value": "There are several ways to use SMB to share files over a local network using a Ubuntu server, one popular way is to use the \"Samba\" software package, which provides file and print services for SMB/CIFS clients..."
      }
    ]
  }
]
```

**Sample Dataset:**
[ShareGPT Samples](https://github.com/fpt-corp/ai-studio-samples/tree/main/sample-datasets/sharegpt)

**Supported file formats:** `.json`, `.jsonlines`, `.zip`, `.parquet`

---

#### 2.2 Trainer = DPO

ShareGPT_DPO contains **conversation samples with ranked responses** (chosen vs rejected) for Direct Preference Optimization (DPO).

**Structure:**

```json
[
   {
      "conversations": [
         { "from": "human", "value": "human message" }
      ],
      "chosen": {
         "from": "gpt",
         "value": "gpt message"
      },
      "rejected": {
         "from": "gpt",
         "value": "gpt message"
      }
   }
]
```

**Example:**

```json
[
  {
    "conversations": [
      {
          "from": "human",
          "value": "Explain if the given argument is an example of correlation or causation and, if appropriate, provide a possible alternative explanation.\nHistory shows that when ice cream sales increase, so do rates of violent crime. Therefore, ice cream causes violent crime."
      },
      {
          "from": "gpt",
          "value": "This is an example of correlation, not causation. A possible alternative explanation could be that ice cream sales and violent crime rates both increase during the warmer months, such as summer, due to higher temperatures or more people being outdoors."
      },
      {
          "from": "human",
          "value": "Discuss the factors that could potentially influence the increase in violent crime rates during warmer months."
      },
      {
          "from": "gpt",
          "value": "Several factors could potentially influence the increase in violent crime rates during warmer months:\n\n1. Increased Outdoor Activity: Warmer weather typically leads to more people spending time outside, which increases the opportunity for interactions, both positive and negative. The increased activity and social interaction can potentially lead to more instances of violent crime..."
      },
      {
          "from": "human",
          "value": "Considering the factors influencing the increase in violent crime rates during warmer months, how might societal and governmental interventions mitigate these effects, particularly focusing on the role of urban planning, education policies, and social services?"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "To mitigate the effects of increased violent crime rates during warmer months through societal and governmental interventions, focusing on urban planning, education policies, and social services, several strategies could be considered:\n\n1. Urban Planning:\n  a. Green spaces: Increasing access to green spaces in urban areas can contribute to a cooling effect and provide a place for residents to relax and recreate, reducing the likelihood of conflict..."
    },
    "rejected": {
      "from": "gpt",
      "value": "Societal and governmental interventions play a critical role in mitigating the effects of factors contributing to an increase in violent crime rates during warmer months. Below are a few examples:\n\n1. Urban Planning: Proper urban planning that promotes green spaces, community parks, and walkable neighborhoods could reduce outdoor tensions, and create opportunities for social interaction, which might decrease the likelihood of violent crime incidents..."
    }
  }
]
```

**Sample Dataset:**
[ShareGPT DPO Samples](https://github.com/fpt-corp/ai-studio-samples/tree/main/sample-datasets/sharegpt-dpo)

**Supported file formats:** `.json`, `.jsonlines`, `.zip`, `.parquet`

---

### 3. **ShareGPT_Image**

An extension of ShareGPT for **multi-modal (text + image)** training. It’s used in fine-tuning vision-language models (VLMs), which need to process images alongside natural language.

The structure includes:
* A list of **chat turns** under **messages**.
* A field called **images** that points to the images used in the conversation (using format png, jpg, jpeg).

**Structure:**

```json
[
   {
      "messages": [
        {"role": "user", "content": "<image> user content"},
        {"role": "assistant", "content": "assistant content"},
      ],
      "images": ["image_path"]
   }
]
```

**Example:**

```json
[
  {
    "messages": [
      { "role": "user", "content": "<image>How many baseball players are visible in the image?" },
      { "role": "assistant", "content": "There are three baseball players visible in the image." }
    ],
    "images": ["images/0.jpg"]
  }
]
```

**Notes:**

* The **`<image>` tokens** must appear in the **user’s contents**.
* **Include the `<image>` token** in the conversation content at the position where an image should appear.
* Image file paths must be specified in the **`images` array**.
* The **placement of images** in the conversation is determined by the **`<image>` tokens**.
* The **number of `<image>` tokens** in the chat must **match the number of entries in the `images` array**.
* Images are **mapped in order of appearance**, with each `<image>` token replaced by its corresponding image from the `images` array.


**Sample Dataset:**
[ShareGPT Image Samples](https://github.com/fpt-corp/ai-studio-samples/tree/main/sample-datasets/sharegpt-image)

**Supported file formats:** `.zip`, `.parquet`

---

### 4. **Corpus**

**Corpus** is a collection of text used for (continual) pre-training language models. Each data point in the corpus includes a **text** field with a string of text.

**Structure:**

```json
[
   {
      "text": "string"
   }
]
```

**Example:**

```json
[
  {
    "text": "Film colorization is any process that adds color to black-and-white images..."
  },
  {
    "text": "In French, gemination is usually not phonologically relevant..."
  }
]
```

**Sample Dataset:**
[Corpus Samples](https://github.com/fpt-corp/ai-studio-samples/tree/main/sample-datasets/corpus)

**Supported file formats:** `.txt`, `.json`, `.jsonlines`, `.zip`, `.parquet`
