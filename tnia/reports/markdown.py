

def image_test(title, report_dir, info_list, figs):
    """ generates a snippet of markdown with a title, a list of information, and a plot

    Args:
        title (string): Title of the test
        info_list (list of strings): A list of info to print in the markdown
        fig (matplotlib figure): figure to print at end of test
    """
    
    markdown='## '+title+' \n\n'

    for info in info_list:
        markdown+=info+' \n\n'

    i=0

    for fig in figs:
        fig_name = title+'_'+str(i)+'.png' 
        fig.savefig(report_dir+fig_name)

        markdown+='![]('+fig_name+')  '
        i=i+1
    markdown+='\n\n'
    
    return markdown

