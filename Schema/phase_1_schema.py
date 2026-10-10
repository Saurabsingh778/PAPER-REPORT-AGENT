from typing import List, Optional
from pydantic import BaseModel, Field

class Result(BaseModel):
    claim: str = Field(description="The headline claim in one sentence.")
    metric: str = Field(description="Metric name, e.g. accuracy, MAE, Pearson r.")
    value: str = Field(description="The number copied EXACTLY as written in the paper, with units.")
    system_or_dataset: str = Field(description="Model/system and dataset this value belongs to.")
    compared_to: Optional[str] = Field(description="Baseline it is compared against, if any.")
    source: str = Field(description="Where it appears, e.g. 'Table 2', 'Fig. 3', 'Abstract'.")

class Link(BaseModel):
    url: str = Field(description="URL exactly as it appears in the candidate list.")
    note: str = Field(description="What this link is (code, dataset, supplementary, other).")

class Extraction(BaseModel):
    main_results: List[Result] = Field(description="At most 10 headline quantitative results.")
    code_repos: List[Link]
    data_links: List[Link]
    availability_statement: Optional[str] = Field(
        description="The paper's data/code availability sentence, copied verbatim, or null.")

