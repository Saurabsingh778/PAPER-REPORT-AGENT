from typing import List, Optional
from pydantic import BaseModel as _PydanticBaseModel, Field

if not hasattr(_PydanticBaseModel, "model_validate"):
    class BaseModel(_PydanticBaseModel):
        def model_dump(self, *args, **kwargs):
            return self.dict(*args, **kwargs)

        def model_dump_json(self, *args, **kwargs):
            return self.json(*args, **kwargs)

        def model_copy(self, *args, **kwargs):
            return self.copy(*args, **kwargs)

        @classmethod
        def model_validate(cls, obj):
            return cls.parse_obj(obj)

        @classmethod
        def model_validate_json(cls, raw):
            return cls.parse_raw(raw)

        @classmethod
        def model_json_schema(cls):
            return cls.schema()
else:
    BaseModel = _PydanticBaseModel


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
        description="The paper's data/code availability sentence, copied verbatim, or null."
    )
