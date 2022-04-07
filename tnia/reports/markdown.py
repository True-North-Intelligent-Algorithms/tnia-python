


def title(title, level=1):

    if level==2:
        markdown='## '
    else:
        markdown='# '

    markdown=markdown+title+' \n\n'
    
    return markdown

def figure(fig, title, fig_title, report_dir):
    fig_name = fig_title+'.png' 
    fig.savefig(report_dir+fig_name)

    markdown = '## '+title+' \n\n'
    markdown+='![]('+fig_name+')  \n\n'

    return markdown

def image_test(title, report_dir, info_list, figs):
    """ generates a snippet of markdown with a title, a list of information, and list of plots

    Args:
        title (string): Title of the test
        info_list (list of strings): A list of info to print in the markdown
        figs (list of matplotlib figures): figures to print at end of test
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

