from pydantic import BaseModel, Field

class PDEOutput(BaseModel):
    install: str = Field(..., description="Python code that installs all required libraries.")
    function: str = Field(..., description="Python code defining the solve_pde function with imports.")
    example: str = Field(..., description="Python code that creates example input and runs solve_pde().")
