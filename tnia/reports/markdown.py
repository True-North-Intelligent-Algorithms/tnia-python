

def image_test(title, report_dir, info_list, fig):
    """ generates a snippet of markdown with a title, a list of information, and a plot

    Args:
        title (string): Title of the test
        info_list (list of strings): A list of info to print in the markdown
        fig (matplotlib figure): figure to print at end of test
    """
    
    markdown='## '+title+' \n\n'

    for info in info_list:
        markdown+=info+' \n\n'

    fig_name = title+'.png' 
    fig.savefig(report_dir+fig_name)

    markdown+='![test image]('+fig_name+')  \n\n'

    return markdown

