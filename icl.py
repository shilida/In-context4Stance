async def select_train_samples(train_df, n_shot, indices):
    system_message = []
    system_message.append(f"""Here are some practical examples:""")
    top10_indices = indices.strip('[]').split(',')
    top10_indices = [int(num) for num in top10_indices]
    i = 1
    for index in top10_indices[:n_shot]:
        try:
            row = train_df.iloc[index]
        except:
            print("no index{}".format(index))
            row = train_df.iloc[0]
        sentence = row['Sentence']
        stance_label = row['Stance']
        topic = row['Target']
        system_message.append(f"""Example: {i}
        Text:{sentence}
        Target:{topic}
        Stance:{stance_label}""")
        i = i + 1
    system_message.append(f"""End of examples.""")
    system_message = "\n".join(system_message)
    return system_message