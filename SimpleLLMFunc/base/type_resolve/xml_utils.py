"""XML utilities for type conversion and example generation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Type, Union, cast, get_origin, get_args
from xml.sax.saxutils import escape

from pydantic import BaseModel


def _is_pydantic_model_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


def _as_pydantic_model_type(value: Any) -> Type[BaseModel]:
    return cast(Type[BaseModel], value)


def _get_root_element_name(type_hint: Any) -> str:
    """根据返回类型智能确定 XML 根元素名称"""
    if _is_pydantic_model_type(type_hint):
        return type_hint.__name__
    return "result"


def _python_to_xml_type(py_type: Any) -> str:
    """将 Python 类型转换为 XML Schema 类型描述"""
    mapping: Dict[Any, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }
    return mapping.get(py_type, "string")


def _convert_value_to_string(value: Any) -> str:
    """将 Python 值转换为 XML 文本内容（处理特殊字符）"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _escape_xml_text(text: str) -> str:
    """转义 XML 文本内容"""
    return escape(text)


def pydantic_to_xml_schema(
    type_hint: Any,
    depth: int = 0,
    max_depth: int = 5,
    seen: Optional[set] = None,
) -> str:
    """生成 Pydantic 模型的 XML Schema 描述文本"""
    from typing import get_origin, get_args, Union as TypingUnion

    if seen is None:
        seen = set()

    if depth > max_depth:
        name = getattr(type_hint, "__name__", str(type_hint))
        return f"{name} (depth limit reached)"

    # Pydantic model
    if _is_pydantic_model_type(type_hint):
        type_id = ("model", type_hint)
        if type_id in seen:
            return f"{type_hint.__name__} (circular reference)"
        seen.add(type_id)

        schema = type_hint.model_json_schema()
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        lines = [f"<{type_hint.__name__}>"]
        model_fields = getattr(type_hint, "model_fields", {})

        for field_name, field_info in properties.items():
            is_required = field_name in required
            req_marker = "required" if is_required else "optional"

            field_ann = None
            if field_name in model_fields:
                field_ann = getattr(model_fields[field_name], "annotation", None)

            field_type_desc = (
                pydantic_to_xml_schema(field_ann, depth + 1, max_depth, seen)
                if field_ann is not None
                else field_info.get("type", "unknown")
            )

            field_desc = field_info.get("description", "")
            extra_info = ""
            if "minimum" in field_info:
                extra_info += f", min: {field_info['minimum']}"
            if "maximum" in field_info:
                extra_info += f", max: {field_info['maximum']}"
            if "default" in field_info:
                extra_info += f", default: {field_info['default']}"

            desc_line = f"  <{field_name}> ({field_type_desc}, {req_marker})"
            if field_desc:
                desc_line += f": {field_desc}"
            if extra_info:
                desc_line += extra_info
            desc_line += f"</{field_name}>"
            lines.append(desc_line)

        lines.append(f"</{type_hint.__name__}>")
        return "\n".join(lines)

    # Get origin after BaseModel check
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # List / sequence
    if origin in (list, List):
        items_type = args[0] if args else Any
        item_desc = pydantic_to_xml_schema(items_type, depth + 1, max_depth, seen)
        return f"<list>\n  <item> ({item_desc})</item>\n</list>"

    # Dict mapping
    if origin in (dict, Dict):
        value_type = args[1] if len(args) >= 2 else Any
        value_desc = pydantic_to_xml_schema(value_type, depth + 1, max_depth, seen)
        return f"<dict>\n  <key> (string)</key>\n  <value> ({value_desc})</value>\n</dict>"

    # Union / Optional
    if origin is TypingUnion:
        non_none = [t for t in args if t is not type(None)]
        if len(non_none) == 1:
            return pydantic_to_xml_schema(non_none[0], depth + 1, max_depth, seen)
        descs = [
            pydantic_to_xml_schema(t, depth + 1, max_depth, seen) for t in non_none
        ]
        return f"Union of: {', '.join(descs)}"

    # Simple types
    return _python_to_xml_type(type_hint)


def _generate_primitive_example(type_hint: Any) -> Any:
    """生成基本类型的示例值"""
    from typing import get_origin, get_args, Union as TypingUnion

    # Handle Optional[T] / Union[T, None]
    origin = get_origin(type_hint)
    if origin is TypingUnion:
        args = get_args(type_hint)
        for t in args:
            if t is not type(None):
                return _generate_primitive_example(t)

    if type_hint is str:
        return "example"
    if type_hint is int:
        return 123
    if type_hint is float:
        return 1.23
    if type_hint is bool:
        return True
    if type_hint is type(None):
        return None

    return None


def generate_xml_example(
    type_hint: Any,
    depth: int = 0,
    max_depth: int = 5,
    seen: Optional[set] = None,
) -> str:
    """生成 XML 格式的示例对象"""
    from typing import get_origin, get_args, Union as TypingUnion

    if seen is None:
        seen = set()

    if depth > max_depth:
        return "<result>...</result>"

    root_name = _get_root_element_name(type_hint)

    # BaseModel
    if _is_pydantic_model_type(type_hint):
        type_id = ("model", type_hint)
        if type_id in seen:
            return f"<{root_name}>...</{root_name}>"
        seen.add(type_id)

        elements = []
        model_fields = getattr(type_hint, "model_fields", {})

        try:
            from pydantic import PydanticUndefined
        except ImportError:
            PydanticUndefined = type("PydanticUndefined", (), {})

        for field_name, field in model_fields.items():
            ann = getattr(field, "annotation", Any)
            default = getattr(field, "default", ...)

            has_default = (
                default is not ...
                and default is not PydanticUndefined
                and not (
                    hasattr(type(default), "__name__")
                    and "PydanticUndefined" in str(type(default))
                )
            )

            if has_default:
                value = default
            else:
                primitive_example = _generate_primitive_example(ann)
                if primitive_example is not None:
                    value = primitive_example
                else:
                    nested_xml = generate_xml_example(ann, depth + 1, max_depth, seen)
                    # 提取嵌套 XML 的内容部分（去掉根标签）
                    if nested_xml.startswith("<") and ">" in nested_xml:
                        # 提取内部内容
                        inner_start = nested_xml.find(">") + 1
                        inner_end = nested_xml.rfind("<")
                        if inner_end > inner_start:
                            inner_content = nested_xml[inner_start:inner_end]
                            elements.append(f"  <{field_name}>{inner_content}</{field_name}>")
                        else:
                            elements.append(f"  <{field_name}>{nested_xml}</{field_name}>")
                    else:
                        elements.append(f"  <{field_name}>{nested_xml}</{field_name}>")
                    continue

            value_str = _escape_xml_text(_convert_value_to_string(value))
            elements.append(f"  <{field_name}>{value_str}</{field_name}>")

        if elements:
            return f"<{root_name}>\n" + "\n".join(elements) + f"\n</{root_name}>"
        return f"<{root_name}></{root_name}>"

    # Get origin after BaseModel check
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # List
    if origin in (list, List):
        item_t = args[0] if args else Any
        item_xml = generate_xml_example(item_t, depth + 1, max_depth, seen)
        # 提取 item 的内容（去掉可能的根标签）
        if item_xml.startswith("<") and ">" in item_xml:
            inner_start = item_xml.find(">") + 1
            inner_end = item_xml.rfind("<")
            if inner_end > inner_start:
                item_content = item_xml[inner_start:inner_end]
            else:
                item_content = item_xml
        else:
            item_content = item_xml

        return f"<{root_name}>\n  <item>{item_content}</item>\n  <item>{item_content}</item>\n</{root_name}>"

    # Dict
    if origin in (dict, Dict):
        val_t = args[1] if len(args) >= 2 else Any
        val_xml = generate_xml_example(val_t, depth + 1, max_depth, seen)
        if val_xml.startswith("<") and ">" in val_xml:
            inner_start = val_xml.find(">") + 1
            inner_end = val_xml.rfind("<")
            if inner_end > inner_start:
                val_content = val_xml[inner_start:inner_end]
            else:
                val_content = val_xml
        else:
            val_content = val_xml

        return f"<{root_name}>\n  <key>example_key</key>\n  <value>{val_content}</value>\n</{root_name}>"

    # Union / Optional
    if origin is TypingUnion:
        for t in args:
            if t is not type(None):
                return generate_xml_example(t, depth + 1, max_depth, seen)
        return f"<{root_name}></{root_name}>"

    # Scalars
    primitive_example = _generate_primitive_example(type_hint)
    if primitive_example is not None:
        value_str = _escape_xml_text(_convert_value_to_string(primitive_example))
        return f"<{root_name}>{value_str}</{root_name}>"

    return f"<{root_name}>example</{root_name}>"


def xml_to_dict(xml_str: str) -> Dict[str, Any]:
    """将 XML 解析为字典（处理 List 的 item 元素）"""
    try:
        root = ET.fromstring(xml_str)
        return _element_to_dict(root, depth=0)
    except ET.ParseError as e:
        raise ValueError(f"无法解析 XML: {str(e)}") from e


def _element_to_dict(element: ET.Element, depth: int = 0) -> Any:
    """将 XML 元素递归转换为字典或值"""
    # 处理属性
    has_attrs = bool(element.attrib)

    children = list(element)
    if children:
        result: Dict[str, Any] = {}
        if has_attrs:
            result.update(element.attrib)

        # 检查是否有多个同名兄弟元素（List 情况）
        child_names = [child.tag for child in children]
        name_counts = {}
        for name in child_names:
            name_counts[name] = name_counts.get(name, 0) + 1

        unique_child_names = set(child_names)
        # 检查是否是包装元素：父元素名是子元素名的复数形式
        # 例如：items -> item, lists -> list, arrays -> array
        child_tag = child_names[0] if child_names else ""
        parent_tag = element.tag
        is_wrapper = (
            len(unique_child_names) == 1
            and name_counts.get(child_tag, 0) > 1
            and (
                (parent_tag.lower().endswith('s') and child_tag.lower() == parent_tag.lower()[:-1])
                or (parent_tag.lower() == child_tag.lower() + 's')
            )
        )
        
        # 特殊处理：包含多个同名子元素的情况
        if (
            not has_attrs
            and len(unique_child_names) == 1
            and name_counts[child_names[0]] > 1
        ):
            child_list = []
            for child in children:
                child_data = _element_to_dict(child, depth + 1)
                child_list.append(child_data)
            
            if depth == 0 and is_wrapper:
                # 根元素且是包装元素：返回字典 {parent_tag: [...]}
                return {parent_tag: child_list}
            elif depth == 0:
                # 根元素但不是包装元素：直接返回列表
                return child_list
            elif is_wrapper:
                # 非根元素但是包装元素：直接返回列表（由父元素处理）
                return child_list

        for child in children:
            child_data = _element_to_dict(child, depth + 1)

            # 检查是否为 List：多个同名兄弟元素
            is_list_item = name_counts.get(child.tag, 0) > 1

            if is_list_item:
                # List 情况：创建列表
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_data)
            else:
                # 单个元素
                # 如果子元素返回的是列表（例如 <items><item>...</item></items>），直接使用
                if isinstance(child_data, list):
                    result[child.tag] = child_data
                elif child.tag in result:
                    # 如果已存在，转换为列表
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = child_data

        return result
    else:
        # 叶子节点：使用文本内容
        text = element.text.strip() if element.text else ""
        if has_attrs:
            # 有属性但无子元素：返回包含属性和文本的字典
            result: Dict[str, Any] = {}
            result.update(element.attrib)
            if text:
                converted = _convert_string_to_type(text)
                result["_text"] = converted if converted is not None else text
            return result
        else:
            # 无属性无子元素：直接返回转换后的值或文本
            if text:
                converted = _convert_string_to_type(text)
                return converted if converted is not None else text
            return ""


def _convert_string_to_type(text: str) -> Any:
    """尝试将字符串转换为合适的 Python 类型"""
    # 布尔值
    if text.lower() in ("true", "false"):
        return text.lower() == "true"

    # 整数
    try:
        return int(text)
    except ValueError:
        pass

    # 浮点数
    try:
        return float(text)
    except ValueError:
        pass

    return None


def dict_to_pydantic(data: Dict[str, Any], model_class: Type[BaseModel]) -> BaseModel:
    """将字典转换为 Pydantic 模型实例"""
    from typing import List as TypingList
    
    # 如果数据是单个值（从 _text 提取），需要包装
    if not isinstance(data, dict):
        # 尝试直接验证
        try:
            return model_class.model_validate({"value": data})
        except Exception:
            return model_class.model_validate(data)

    # 获取模型字段的类型注解
    model_fields = getattr(model_class, "model_fields", {})
    field_annotations = {}
    for field_name, field_info in model_fields.items():
        field_annotations[field_name] = getattr(field_info, "annotation", None)

    # 清理 _text 键，将其内容提升到父级，并处理列表格式
    cleaned_data = {}
    for key, value in data.items():
        if key == "_text":
            # 如果只有一个 _text，尝试直接验证
            if len(data) == 1:
                try:
                    return model_class.model_validate(value)
                except Exception:
                    pass
            cleaned_data[key] = value
        elif isinstance(value, dict) and "_text" in value and len(value) == 1:
            # 提升 _text 内容
            cleaned_data[key] = value["_text"]
        elif isinstance(value, dict) and "item" in value:
            # 处理 {'item': [...]} 或 {'item': value} 格式
            field_type = field_annotations.get(key)
            if field_type:
                origin = get_origin(field_type)
                if origin in (list, TypingList):
                    # 列表类型
                    args = get_args(field_type)
                    item_type = args[0] if args else Any
                    
                    if isinstance(value["item"], list):
                        # 多个 item：{'item': [...]}
                        item_list = value["item"]
                        # 如果列表元素是 Pydantic 模型，递归转换
                        if _is_pydantic_model_type(item_type):
                            cleaned_data[key] = [
                                dict_to_pydantic(item, _as_pydantic_model_type(item_type)) if isinstance(item, dict) else item
                                for item in item_list
                            ]
                        else:
                            cleaned_data[key] = item_list
                    else:
                        # 单个 item：{'item': value}，包装成列表
                        single_item = value["item"]
                        if _is_pydantic_model_type(item_type):
                            cleaned_data[key] = [
                                dict_to_pydantic(single_item, _as_pydantic_model_type(item_type))
                                if isinstance(single_item, dict)
                                else single_item
                            ]
                        else:
                            cleaned_data[key] = [single_item]
                else:
                    cleaned_data[key] = value
            else:
                cleaned_data[key] = value
        elif isinstance(value, list):
            # 递归处理列表中的字典
            field_type = field_annotations.get(key)
            if field_type:
                args = get_args(field_type)
                item_type = args[0] if args else Any
                # 如果列表元素是 Pydantic 模型，递归转换
                if _is_pydantic_model_type(item_type):
                    cleaned_data[key] = [
                        dict_to_pydantic(item, _as_pydantic_model_type(item_type)) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    cleaned_data[key] = [
                        item["_text"] if isinstance(item, dict) and "_text" in item and len(item) == 1 else item
                        for item in value
                    ]
            else:
                cleaned_data[key] = [
                    item["_text"] if isinstance(item, dict) and "_text" in item and len(item) == 1 else item
                    for item in value
                ]
        elif isinstance(value, dict):
            # 检查字段类型注解
            field_type = field_annotations.get(key)
            
            # 先检查是否为列表类型（{'item': [...]} 格式）
            if field_type:
                origin = get_origin(field_type)
                if origin in (list, TypingList) and "item" in value and isinstance(value["item"], list):
                    # 处理 {'item': [...]} 格式的列表
                    item_list = value["item"]
                    args = get_args(field_type)
                    item_type = args[0] if args else Any
                    
                    # 如果列表元素是 Pydantic 模型，递归转换
                    if _is_pydantic_model_type(item_type):
                        cleaned_data[key] = [
                            dict_to_pydantic(item, _as_pydantic_model_type(item_type)) if isinstance(item, dict) else item
                            for item in item_list
                        ]
                    else:
                        cleaned_data[key] = item_list
                elif field_type and _is_pydantic_model_type(field_type):
                    # 递归转换嵌套的 Pydantic 模型
                    cleaned_data[key] = dict_to_pydantic(value, _as_pydantic_model_type(field_type))
                else:
                    cleaned_data[key] = value
            else:
                cleaned_data[key] = value
        else:
            # 基本类型转换：根据字段类型注解进行转换
            field_type = field_annotations.get(key)
            if field_type:
                # 处理 Optional 类型
                origin = get_origin(field_type)
                if origin is Union:
                    args = get_args(field_type)
                    non_none_types = [t for t in args if t is not type(None)]
                    if non_none_types:
                        field_type = non_none_types[0]
                        origin = get_origin(field_type)
                
                # 类型转换
                if field_type is str and not isinstance(value, str):
                    cleaned_data[key] = str(value)
                elif field_type is int and not isinstance(value, int):
                    try:
                        cleaned_data[key] = int(value)
                    except (ValueError, TypeError):
                        cleaned_data[key] = value
                elif field_type is float and not isinstance(value, float):
                    try:
                        cleaned_data[key] = float(value)
                    except (ValueError, TypeError):
                        cleaned_data[key] = value
                elif field_type is bool and not isinstance(value, bool):
                    if isinstance(value, str):
                        cleaned_data[key] = value.lower() in ("true", "yes", "1")
                    else:
                        cleaned_data[key] = bool(value)
                else:
                    cleaned_data[key] = value
            else:
                cleaned_data[key] = value

    try:
        return model_class.model_validate(cleaned_data)
    except Exception as e:
        # 如果验证失败，尝试使用原始数据
        try:
            return model_class.model_validate(data)
        except Exception:
            raise ValueError(f"无法将字典转换为 Pydantic 模型: {str(e)}") from e
