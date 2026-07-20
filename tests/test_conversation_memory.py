import pytest
from unittest.mock import patch, MagicMock
from tools.memory import (
    _is_permanent,
    search_conversation_memory,
    cleanup_expired_conversation_memory,
)

# _is_permanent 测试
def test_permanent_diabetes():
    assert _is_permanent("用户有糖尿病，需要低糖饮食") == True

def test_permanent_hypertension():
    assert _is_permanent("用户高血压，推荐低盐食谱") == True

def test_permanent_allergy():
    assert _is_permanent("用户对海鲜过敏") == True

def test_not_permanent_normal():
    assert _is_permanent("用户想吃番茄炒蛋，推荐了家常菜") == False

def test_not_permanent_price():
    assert _is_permanent("用户查询苹果价格，Woolworths $3.5/kg") == False

# search 空结果不报错
def test_search_returns_list():
    result = search_conversation_memory("nonexistent_user_xyz", "随便问点什么")
    assert isinstance(result, list)

# cleanup 空库不报错
def test_cleanup_no_error():
    cleanup_expired_conversation_memory()  # 不抛异常即通过

# permanent 记录不被清理（通过 _is_permanent 间接验证）
def test_permanent_gout():
    assert _is_permanent("用户痛风发作期，禁止高嘌呤食物") == True
