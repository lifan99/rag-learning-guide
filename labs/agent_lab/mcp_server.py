"""Read-only MCP tools for a locally authenticated teaching session."""
from mcp.server import MCPServer
from pydantic import BaseModel
from agent_lab import POLICY, employee_facts

server = MCPServer("training-knowledge")
# The local host supplies identity; a model cannot select another employee.
AUTHENTICATED_EMPLOYEE = "employee001"


class PolicyResult(BaseModel):
    source: str
    minimum_months: int
    preapproval_required: bool
    annual_limit_yuan: int


class EmployeeResult(BaseModel):
    source: str
    months: int
    preapproved: bool
    used_yuan: int


@server.tool()
def lookup_policy() -> PolicyResult:
    """Read the current training subsidy policy and its source version."""
    return PolicyResult(**POLICY)


@server.tool()
def lookup_employee() -> EmployeeResult:
    """Read facts of the employee authenticated by this local host."""
    return EmployeeResult(**employee_facts(AUTHENTICATED_EMPLOYEE))


if __name__ == "__main__":
    server.run(transport="stdio")
