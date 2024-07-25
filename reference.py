import fitz

### READ IN PDF
doc = fitz.open("./Docs/stToPal.pdf")

for page in doc:
    ### SEARCH
    text = '''
As per RBI guidelines, the refund of Ticket should be given in the same Bank account, which was used for booking. It is necessary that the Bank 
Account used for booking online ticket should not be closed at least up to 30 days beyond the date of the journey. If accounts are found closed at 
the time of processing refund, the refund will be regretted by the Bank.
''' 
    text_instances = page.search_for(text)

    ### HIGHLIGHT
    for inst in text_instances:
        highlight = page.add_highlight_annot(inst)
        highlight.update()


### OUTPUT
doc.save("output.pdf", garbage=4, deflate=True, clean=True)


