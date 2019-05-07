from django import template

register = template.Library()

@register.filter
def previous(some_list, current_index):
    """
    Returns the previous element of the list using the current index if it exists.
    Otherwise returns an empty string.
    """
    if not current_index:
        return ''
    else:
        return some_list[int(current_index) - 1]
