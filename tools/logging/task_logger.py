"""
任务日志推送工具 - 用于后端向前端推送任务进度和日志信息
"""
import requests
from typing import Literal, Optional


class TaskLogger:
    """任务日志推送器"""
    
    def __init__(self, ui_url: str = "http://127.0.0.1:8000"):
        """
        初始化日志推送器
        
        Args:
            ui_url: UI服务的URL，默认为本机8000端口
        """
        self.ui_url = ui_url
        self.log_endpoint = f"{ui_url}/api/task/log"
    
    def log(self, message: str, level: Literal["info", "success", "warning", "error"] = "info") -> bool:
        """
        推送一条日志到前端
        
        Args:
            message: 日志内容
            level: 日志级别，可选值: info(蓝), success(绿), warning(橙), error(红)
        
        Returns:
            是否成功推送
        """
        try:
            resp = requests.post(
                self.log_endpoint,
                json={"message": message, "level": level},
                timeout=2
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"[日志推送失败] {e}")
            return False
    
    def info(self, message: str) -> bool:
        """推送info级别日志"""
        return self.log(message, "info")
    
    def success(self, message: str) -> bool:
        """推送success级别日志"""
        return self.log(message, "success")
    
    def warning(self, message: str) -> bool:
        """推送warning级别日志"""
        return self.log(message, "warning")
    
    def error(self, message: str) -> bool:
        """推送error级别日志"""
        return self.log(message, "error")


# 全局日志实例
_logger_instance: Optional[TaskLogger] = None


def get_logger(ui_url: str = "http://127.0.0.1:8000") -> TaskLogger:
    """
    获取全局日志实例（单例模式）
    
    Args:
        ui_url: UI服务的URL
    
    Returns:
        TaskLogger实例
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TaskLogger(ui_url)
    return _logger_instance


# 简便的快速调用函数
def log(message: str, level: Literal["info", "success", "warning", "error"] = "info") -> bool:
    """推送日志"""
    return get_logger().log(message, level)


def log_info(message: str) -> bool:
    """推送info级别日志"""
    return get_logger().info(message)


def log_success(message: str) -> bool:
    """推送success级别日志"""
    return get_logger().success(message)


def log_warning(message: str) -> bool:
    """推送warning级别日志"""
    return get_logger().warning(message)


def log_error(message: str) -> bool:
    """推送error级别日志"""
    return get_logger().error(message)


def log_debug(message: str) -> bool:
    """Debug级别暂映射到info"""
    return get_logger().info(f"[DEBUG] {message}")
