from fastapi import Header, HTTPException

import app.services.session_manager as sm_mod
from app.services.session_manager import Session

SESSION_TOKEN_HEADER = "X-Session-Token"


def require_session_access(
    session_id: str,
    session_token: str,
) -> Session:
    session = sm_mod.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not sm_mod.session_manager.verify_access_token(session_id, session_token):
        raise HTTPException(status_code=401, detail="Invalid session token")

    return session


def session_token_header(
    session_token: str = Header(..., alias=SESSION_TOKEN_HEADER),
) -> str:
    return session_token
